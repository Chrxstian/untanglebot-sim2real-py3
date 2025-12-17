#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision
from torchvision import models
from . import config

class TangledRopeKeypointModel(nn.Module):
    def conv_block(self, in_c, out_c, kernel=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def up_conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

    def _add_coord_channels(self, x):
        batch_size, _, height, width = x.shape
        
        # Create x and y grids
        xx_channel = torch.arange(width, dtype=x.dtype, device=x.device).view(1, 1, 1, width).expand(batch_size, 1, height, width)
        yy_channel = torch.arange(height, dtype=x.dtype, device=x.device).view(1, 1, height, 1).expand(batch_size, 1, height, width)
        
        # Normalize coordinates to range [-1, 1]
        xx_channel = (xx_channel / (width - 1)) * 2 - 1
        yy_channel = (yy_channel / (height - 1)) * 2 - 1
        
        return torch.cat([x, xx_channel, yy_channel], dim=1)

    def _create_pixel_head(self, in_channels, out_channels, activation=None):
        # Pixel-wise prediction head (uses CoordConv input)
        input_dim = in_channels + 2 
        
        layers = [
            nn.Conv2d(input_dim, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, kernel_size=1)
        ]
        
        if activation:
            layers.append(activation)
            
        return nn.Sequential(*layers)
    
    def _create_coord_head(self, in_channels):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, 16),
            nn.ReLU(),
            nn.Linear(16, 2) 
        )
    
    def _create_type_head(self, in_features_flat):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features_flat, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)  
        )

    def __init__(self):
        super(TangledRopeKeypointModel, self).__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        target_in_channels = config.IMAGE_CHANNELS # 3
        
        old_weights = resnet.conv1.weight.data # Shape (64, 3, 7, 7)
        
        resnet.conv1 = nn.Conv2d(target_in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if target_in_channels == 3:
            avg_weight = torch.mean(old_weights, dim=1, keepdim=True) # Shape (64, 1, 7, 7)
            resnet.conv1.weight.data = avg_weight.repeat(1, 3, 1, 1)
        else:
            # Fallback
            resnet.conv1.weight.data[:, :3, :, :] = old_weights
            if target_in_channels > 3:
                avg = torch.mean(old_weights, dim=1, keepdim=True)
                resnet.conv1.weight.data[:, 3:, :, :] = avg.repeat(1, target_in_channels-3, 1, 1)

        # --- Encoder ---
        self.encoder_layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder_maxpool = resnet.maxpool
        self.encoder_layer1 = resnet.layer1 
        self.encoder_layer2 = resnet.layer2 
        self.encoder_layer3 = resnet.layer3 
        self.encoder_layer4 = resnet.layer4 

        # --- Decoder ---
        self.dec_up1 = self.up_conv_block(512, 256)
        self.dec_conv1 = self.conv_block(256 + 256, 256) 

        self.dec_up2 = self.up_conv_block(256, 128)
        self.dec_conv2 = self.conv_block(128 + 128, 128) 

        self.dec_up3 = self.up_conv_block(128, 64)
        self.dec_conv3 = self.conv_block(64 + 64, 64)   
        
        self.dec_up4 = self.up_conv_block(64, 64)
        self.dec_conv4 = self.conv_block(64 + 64, 32)   

        self.final_up = self.up_conv_block(32, 16)
        self.final_conv = self.conv_block(16 + target_in_channels, 16)

        # --- Heads ---
        self.endpoint_l_head = self._create_coord_head(in_channels=17)
        self.endpoint_r_head = self._create_coord_head(in_channels=17)
        self.intersec_l_head = self._create_coord_head(in_channels=17)
        self.intersec_r_head = self._create_coord_head(in_channels=17)

        self.input_channels_type = 18 
        pooled_feature_count = self.input_channels_type * config.ROI_POOL_SIZE * config.ROI_POOL_SIZE
        
        self.intersec_type_l_head = self._create_type_head(pooled_feature_count)
        self.intersec_type_r_head = self._create_type_head(pooled_feature_count)

        # Pixel-wise Heads (Length & Direction)
        self.length_map_head = self._create_pixel_head(in_channels=16, out_channels=2, activation=nn.Sigmoid())
        self.direction_map_head = self._create_pixel_head(in_channels=16, out_channels=4, activation=nn.Tanh())

    def forward(self, x):
        # Encoder
        skip0 = self.encoder_layer0(x)         
        skip1 = self.encoder_maxpool(skip0)
        skip1 = self.encoder_layer1(skip1)     
        skip2 = self.encoder_layer2(skip1)     
        skip3 = self.encoder_layer3(skip2)     
        bottleneck = self.encoder_layer4(skip3)

        # Decoder
        d1 = self.dec_up1(bottleneck)
        d1 = torch.cat([d1, skip3], dim=1)
        d1 = self.dec_conv1(d1)

        d2 = self.dec_up2(d1)
        d2 = torch.cat([d2, skip2], dim=1)
        d2 = self.dec_conv2(d2)

        d3 = self.dec_up3(d2)
        d3 = torch.cat([d3, skip1], dim=1)
        d3 = self.dec_conv3(d3)

        d4 = self.dec_up4(d3)
        d4 = torch.cat([d4, skip0], dim=1)
        d4 = self.dec_conv4(d4)

        d5 = self.final_up(d4)
        master_features = self.final_conv(torch.cat([d5, x], dim=1)) 

        # --- Pixel Heads (Length & Direction) ---
        features_with_coords = self._add_coord_channels(master_features)
        
        length_maps = self.length_map_head(features_with_coords) 
        direction_maps = self.direction_map_head(features_with_coords)

        # --- Coordinate & Type Heads ---
        # Mask logic: 
        # Channel 0 = Left Rope Input (Distance Transform)
        # Channel 1 = Right Rope Input (Distance Transform)
        left_rope_mask = x[:, 0:1, :, :]
        right_rope_mask = x[:, 1:2, :, :]
        
        features_for_left_coords = torch.cat([master_features, left_rope_mask], dim=1)
        features_for_right_coords = torch.cat([master_features, right_rope_mask], dim=1)

        features_for_left_type = torch.cat([master_features, left_rope_mask, right_rope_mask], dim=1)
        features_for_right_type = torch.cat([master_features, right_rope_mask, left_rope_mask], dim=1)

        coords_endpoint_l = self.endpoint_l_head(features_for_left_coords)
        coords_endpoint_r = self.endpoint_r_head(features_for_right_coords)
        coords_intersec_l = self.intersec_l_head(features_for_left_coords)
        coords_intersec_r = self.intersec_r_head(features_for_right_coords)

        # RoI Logic for Types
        rois_l_coords = coords_intersec_l
        rois_r_coords = coords_intersec_r
        box_radius = config.BOX_RADIUS
        
        boxes_l = torch.cat([
            rois_l_coords[:, 1:2] - box_radius,
            rois_l_coords[:, 0:1] - box_radius,
            rois_l_coords[:, 1:2] + box_radius,
            rois_l_coords[:, 0:1] + box_radius 
        ], dim=1)
        
        boxes_r = torch.cat([
            rois_r_coords[:, 1:2] - box_radius,
            rois_r_coords[:, 0:1] - box_radius,
            rois_r_coords[:, 1:2] + box_radius,
            rois_r_coords[:, 0:1] + box_radius 
        ], dim=1)

        batch_size = x.shape[0]
        batch_idx = torch.arange(batch_size, device=x.device).float().view(-1, 1)

        rois_l = torch.cat([batch_idx, boxes_l], dim=1)
        rois_r = torch.cat([batch_idx, boxes_r], dim=1)

        pooled_features_l = torchvision.ops.roi_align(features_for_left_type, rois_l, output_size=(config.ROI_POOL_SIZE, config.ROI_POOL_SIZE), spatial_scale=1.0)
        pooled_features_r = torchvision.ops.roi_align(features_for_right_type, rois_r, output_size=(config.ROI_POOL_SIZE, config.ROI_POOL_SIZE), spatial_scale=1.0)
        
        type_l_logit = self.intersec_type_l_head(pooled_features_l)
        type_r_logit = self.intersec_type_r_head(pooled_features_r)
        
        coordinates = torch.stack([coords_endpoint_l, coords_intersec_l, coords_endpoint_r, coords_intersec_r], dim=1)
        types = torch.cat([type_l_logit, type_r_logit], dim=1)        

        return {
            'coordinates': coordinates, 
            'types': types, 
            'length_maps': length_maps, 
            'direction_maps': direction_maps
        }
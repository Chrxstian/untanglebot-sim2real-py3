#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import roslibpy
import time
import roslibpy.core
import sys
import json

ROS_BRIDGE_HOST = 'localhost'
ROS_BRIDGE_PORT = 9090
SERVICE_NAME = '/policy_action_service'
SERVICE_TYPE = 'untanglebot/PolicyAction'

TEST_CASES = {
    '1': {
        "name": "NO-OP (Both -1)",
        "request": {
            'force_left': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_left': -1,
            'force_right': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_right': -1,
            'frame_id': 'WAIST'
        }
    },
    # --- Unimanual Tests ---
    '2': {
            "name": "ONE HAND PULL +x",
            "request": {
                'force_left': {'x': 1.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_left': 20,
                'force_right': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_right': -1,
                'frame_id': 'WAIST'
            }
        },
    '3': {
            "name": "ONE HAND PULL -x",
            "request": {
                'force_left': {'x': -1.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_left': 20,
                'force_right': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_right': -1,
                'frame_id': 'WAIST'
            }
        },
    '4': {
            "name": "ONE HAND PULL +y",
            "request": {
                'force_left': {'x': 0.0, 'y': 1.0, 'z': 0.0}, 'grasp_idx_left': 20,
                'force_right': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_right': -1,
                'frame_id': 'WAIST'
            }
        },
    '5': {
            "name": "ONE HAND PULL -y",
            "request": {
                'force_left': {'x': 0.0, 'y': -1.0, 'z': 0.0}, 'grasp_idx_left': 20,
                'force_right': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_right': -1,
                'frame_id': 'WAIST'
            }
        },
    '6': {
        "name": "ONE HAND LIFT red",
        "request": {
            'force_left': {'x': 0.0, 'y': 0.0, 'z': 1.0}, 'grasp_idx_left': 20,
            'force_right': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_right': -1,
            'frame_id': 'WAIST'
        }
    },
    '7': {
        "name": "ONE HAND LIFT pink",
        "request": {
            'force_left': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_left': -1,
            'force_right': {'x': 0.0, 'y': 0.0, 'z': 1.0}, 'grasp_idx_right': 20,
            'frame_id': 'WAIST'
        }
    },
    # --- Bimanual Tests ---
    '8': {
        "name": "Bimanual LIFT & PULL",
        "request": {
            'force_left': {'x': 0.0, 'y': 0.0, 'z': 1.0}, 'grasp_idx_left': 20,
            'force_right': {'x': 1.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_right': 20,
            'frame_id': 'WAIST'
        }
    },
    '9': {
        "name": "Bimanual PIN LEFT & PULL RIGHT",
        "request": {
            'force_left': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_left': 20,
            'force_right': {'x': 0.0, 'y': 1.0, 'z': 0.0}, 'grasp_idx_right': 20,
            'frame_id': 'WAIST'
        }
    },
    '10': {
        "name": "Bimanual PIN RIGHT & PULL LEFT",
        "request": {
            'force_left': {'x': 0.0, 'y': -1.0, 'z': 0.0}, 'grasp_idx_left': 20,
            'force_right': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_right': 20,
            'frame_id': 'WAIST'
        }
    },
    '11': {
        "name": "Bimanual PULL & PULL",
        "request": {
            'force_left': {'x': 0.0, 'y': 1.0, 'z': 0.0}, 'grasp_idx_left': 20,
            'force_right': {'x': 1.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_right': 20,
            'frame_id': 'WAIST'
        }
    },
    '12': {
        "name": "Bimanual PIN & PIN",
        "request": {
            'force_left': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_left': 20,
            'force_right': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_right': 20,
            'frame_id': 'WAIST'
        }
    }
}

def call_service(client, service, test_name, request_data):
    """
    Calls the service with the given request data.
    """
    try:
        request = roslibpy.ServiceRequest(request_data)

        print("\n" + "="*50)
        print("EXECUTING TEST: %s" % test_name)
        print("Sending request:")
        print(json.dumps(request_data, indent=2))

        # Call the service and wait for the response
        result = service.call(request)

        print("\nReceived service response:")
        print(result)
        print("="*50 + "\n")
        
        # Give some time for the robot to finish moving before next command
        time.sleep(1.0) 

    except roslibpy.core.RosTimeoutError as e:
        print("SERVICE CALL FAILED: Timeout.")
        print(e)
    except Exception as e:
        print("SERVICE CALL FAILED: %s" % e)


def print_menu():
    """Prints the interactive test menu."""
    print("\n--- Policy Action Bridge Test Client ---")
    # Dynamically print menu items from the dictionary
    for key, value in sorted(TEST_CASES.items(), key=lambda item: int(item[0])):
        print(" %s. %s" % (key, value["name"]))
    print("----------------------------------------")
    all_key = str(len(TEST_CASES) + 1)
    print(" %s. Run ALL tests (1-%d)" % (all_key, len(TEST_CASES)))
    print(" 0. Quit")
    print("----------------------------------------")
    
    return input("Enter your choice (0-%s): " % all_key)

def main():
    """
    Main function to connect to ROS and handle the test menu.
    """
    client = roslibpy.Ros(host=ROS_BRIDGE_HOST, port=ROS_BRIDGE_PORT)
    
    try:
        print("Attempting to connect to rosbridge at %s:%s..." % (ROS_BRIDGE_HOST, ROS_BRIDGE_PORT))
        client.run()
        
        connect_timeout = 10 # seconds
        start_time = time.time()
        while not client.is_connected:
            time.sleep(0.1)
            if time.time() - start_time > connect_timeout:
                raise Exception("Connection to rosbridge timed out.")
        
        print("Successfully connected to rosbridge.")

        # Define the service client once
        service = roslibpy.Service(client, SERVICE_NAME, SERVICE_TYPE)
        
        all_key = str(len(TEST_CASES) + 1)

        # Main test loop
        while True:
            choice = print_menu()
            
            if choice == '0':
                print("Quitting.")
                break
            
            elif choice == all_key:
                print("--- RUNNING ALL TESTS (1-%d) ---" % len(TEST_CASES))
                for key in sorted(TEST_CASES.keys(), key=lambda k: int(k)):
                    test = TEST_CASES[key]
                    call_service(client, service, test["name"], test["request"])
                print("--- ALL TESTS COMPLETE ---")

            elif choice in TEST_CASES:
                test = TEST_CASES[choice]
                call_service(client, service, test["name"], test["request"])
            
            else:
                print("Invalid choice. Please try again.")

    except Exception as e:
        print("An error occurred: %s" % e)
    finally:
        if client.is_connected:
            try:
                client.terminate()
                print("Connection terminated.")
            except AttributeError as e:
                print(f"roslibpy internal error during terminate (ignoring): {e}")

if __name__ == '__main__':
    main()
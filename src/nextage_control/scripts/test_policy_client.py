#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import time
import json

from nextage_control.srv import ExecutePolicyAction, ExecutePolicyActionRequest

SERVICE_NAME = '/execute_policy_action'

TEST_CASES = {
    '1': {
        "name": "NO-OP (Both -1)",
        "request": {
            'force_left': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_left': -1,
            'force_right': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_right': -1,
            'frame_id': 'WAIST', 'force_threshold': 15.0
        }
    },
    # --- Unimanual Tests ---
    '2': {
            "name": "ONE HAND PULL +x",
            "request": {
                'force_left': {'x': 1.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_left': 20,
                'force_right': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_right': -1,
                'frame_id': 'WAIST', 'force_threshold': 15.0
            }
        },
    '3': {
            "name": "ONE HAND PULL -x",
            "request": {
                'force_left': {'x': -1.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_left': 20,
                'force_right': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_right': -1,
                'frame_id': 'WAIST', 'force_threshold': 15.0
            }
        },
    '4': {
            "name": "ONE HAND PULL +y",
            "request": {
                'force_left': {'x': 0.0, 'y': 1.0, 'z': 0.0}, 'grasp_idx_left': 20,
                'force_right': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_right': -1,
                'frame_id': 'WAIST', 'force_threshold': 15.0
            }
        },
    '5': {
            "name": "ONE HAND PULL -y",
            "request": {
                'force_left': {'x': 0.0, 'y': -1.0, 'z': 0.0}, 'grasp_idx_left': 20,
                'force_right': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_right': -1,
                'frame_id': 'WAIST', 'force_threshold': 15.0
            }
        },
    '6': {
        "name": "ONE HAND LIFT red",
        "request": {
            'force_left': {'x': 0.0, 'y': 0.0, 'z': 1.0}, 'grasp_idx_left': 20,
            'force_right': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_right': -1,
            'frame_id': 'WAIST', 'force_threshold': 15.0
        }
    },
    '7': {
        "name": "ONE HAND LIFT pink",
        "request": {
            'force_left': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_left': -1,
            'force_right': {'x': 0.0, 'y': 0.0, 'z': 1.0}, 'grasp_idx_right': 20,
            'frame_id': 'WAIST', 'force_threshold': 15.0
        }
    },
    # --- Bimanual Tests ---
    '8': {
        "name": "Bimanual LIFT & PULL",
        "request": {
            'force_left': {'x': 0.0, 'y': 0.0, 'z': 1.0}, 'grasp_idx_left': 20,
            'force_right': {'x': 1.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_right': 20,
            'frame_id': 'WAIST', 'force_threshold': 15.0
        }
    },
    '9': {
        "name": "Bimanual PIN LEFT & PULL RIGHT",
        "request": {
            'force_left': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_left': 13,
            'force_right': {'x': 0.0, 'y': 1.0, 'z': 0.0}, 'grasp_idx_right': 18,
            'frame_id': 'WAIST', 'force_threshold': 15.0
        }
    },
    '10': {
        "name": "Bimanual PIN RIGHT & PULL LEFT",
        "request": {
            'force_left': {'x': 0.0, 'y': -1.0, 'z': 0.0}, 'grasp_idx_left': 13,
            'force_right': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_right': 18,
            'frame_id': 'WAIST', 'force_threshold': 15.0
        }
    },
    '11': {
        "name": "Bimanual PULL & PULL",
        "request": {
            'force_left': {'x': 0.0, 'y': 1.0, 'z': 0.0}, 'grasp_idx_left': 20,
            'force_right': {'x': 1.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_right': 20,
            'frame_id': 'WAIST', 'force_threshold': 15.0
        }
    },
    '12': {
        "name": "Bimanual PIN & PIN",
        "request": {
            'force_left': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_left': 20,
            'force_right': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'grasp_idx_right': 20,
            'frame_id': 'WAIST', 'force_threshold': 15.0
        }
    }
}

def call_service(service_proxy, test_name, request_data):
    """
    Calls the service with the given request data.
    """
    try:
        req = ExecutePolicyActionRequest()
        
        req.Force_left.x = request_data['force_left']['x']
        req.Force_left.y = request_data['force_left']['y']
        req.Force_left.z = request_data['force_left']['z']
        req.grasp_idx_left = int(request_data['grasp_idx_left'])
        
        req.Force_right.x = request_data['force_right']['x']
        req.Force_right.y = request_data['force_right']['y']
        req.Force_right.z = request_data['force_right']['z']
        req.grasp_idx_right = int(request_data['grasp_idx_right'])
        
        req.frame = request_data['frame_id']
        req.force_threshold = float(request_data['force_threshold'])

        print("\n" + "="*50)
        print("EXECUTING TEST: %s" % test_name)
        print("Sending request:")
        print(json.dumps(request_data, indent=2))

        # Call the service and wait for the response
        result = service_proxy(req)

        print("\nReceived service response:")
        print(result)
        print("="*50 + "\n")
        
        # Give some time for the robot to finish moving before next command
        time.sleep(1.0) 

    except rospy.ServiceException as e:
        print("SERVICE CALL FAILED: %s" % e)
    except Exception as e:
        print("AN ERROR OCCURRED: %s" % e)


def print_menu():
    """Prints the interactive test menu."""
    print("\n--- Policy Action Bridge Test Client ---")
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
    rospy.init_node('test_policy_client', anonymous=True)
    
    try:
        print(f"Waiting for service: {SERVICE_NAME} ...")
        rospy.wait_for_service(SERVICE_NAME, timeout=10.0)
        print("Service found.")

        # Create Service Proxy
        service_proxy = rospy.ServiceProxy(SERVICE_NAME, ExecutePolicyAction)
        
        all_key = str(len(TEST_CASES) + 1)

        # Main test loop
        while not rospy.is_shutdown():
            choice = print_menu()
            
            if choice == '0':
                print("Quitting.")
                break
            
            elif choice == all_key:
                print("--- RUNNING ALL TESTS (1-%d) ---" % len(TEST_CASES))
                for key in sorted(TEST_CASES.keys(), key=lambda k: int(k)):
                    test = TEST_CASES[key]
                    call_service(service_proxy, test["name"], test["request"])
                print("--- ALL TESTS COMPLETE ---")

            elif choice in TEST_CASES:
                test = TEST_CASES[choice]
                call_service(service_proxy, test["name"], test["request"])
            
            else:
                print("Invalid choice. Please try again.")

    except rospy.ROSException as e:
        print(f"ROS Error (Timeout?): {e}")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print("An error occurred: %s" % e)

if __name__ == '__main__':
    main()
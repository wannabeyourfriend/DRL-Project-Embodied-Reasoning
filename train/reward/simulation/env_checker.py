import os
import sys
import math
import copy

# Ensure the project root is in sys.path
project_root1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
project_root2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
project_root3 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root1 not in sys.path:
    sys.path.insert(0, project_root1)
if project_root2 not in sys.path:
    sys.path.insert(0, project_root2)
if project_root3 not in sys.path:
    sys.path.insert(0, project_root3)

# These imports should work if you run the script within your project structure.
# Mocks are provided at the end for standalone testing.
from embodied_reasoner.evaluate.ai2thor_engine.RocAgent import RocAgent
from ai2thor.controller import Controller
from embodied_reasoner.api_keys_config import QWEN_API_KEY

class EnvChecker:
    def __init__(self, env_config=None):
        self.max_steps = env_config.get('max_steps', 20)
        self.task = env_config.get('task', {})
        controller = Controller(
            scene=env_config.get('scene', 'FloorPlan203'),
            gridSize=0.25,
            width=640,
            height=480,
            fieldOfView=90,
            renderDepthImage=True
        )
        self.agent = RocAgent(
            controller=controller, 
            save_path=None,
            scene=env_config.get('scene', 'FloorPlan203'),
            visibilityDistance=1.5,
            gridSize=0.25, 
            fieldOfView=90,
            target_objects=env_config.get("target_objects", ["RemoteControl|1", "CoffeeTable|1"]), 
            related_objects=env_config.get("related_objects", ["CoffeeTable|1"]),
            navigable_objects=env_config.get("navigable_objects", ["CounterTop", "Sink", "Fridge", "DiningTable", "Chair", "CoffeeTable", "Sofa", "TVStand"]),
            taskid=env_config.get("task_id", "0"),
            platform_type=env_config.get("platform_type", "GPU")
        )
        
        # tool variables
        self.metadata = None
        self.event = None
        self.navigable_list = []
    
    def check(self, plan):
        """
        Check if the plan can be executed in the environment.
        This method should be implemented to interact with the AI2THOR environment.
        """
        info = {}
        info["step"] = 0
        info["success"] = False
        
        # first initialize the agent in the corner and then observe the environment
        self.reward = 0
        self.wrong_time = 0
        action_result = self.agent.init_agent_corner()
        action_result = self.agent.observe()
        
        for step in range(self.max_steps):
            self.update()
            
            # get action and objectID
            decision_making = plan[step]
            action, object_name = self.split_decision(decision_making)
            if object_name is None:
                objectId = None
            else:
                objectType = object_name
                match_ids = [
                    obj['objectId']
                    for obj in self.event.metadata['objects']
                    if obj['objectType'] == objectType
                ]
                if len(match_ids) >= 2:
                    print(f"More than 1 object found with type '{objectType}'")
                    continue
                elif len(match_ids) == 0:
                    print(f"No object found with type '{objectType}'")
                    continue
                objectId = match_ids[0]

            # analyze the action and object
            if action == "end": # Task marked as 'end' by the model
                self.plan_end = True
            elif action == "observe":
                action_result = self.agent.observe()
            elif action == "move forward":
                action_result = self.agent.move_forward(0.5)
            elif object_name:
                if action == "navigate to": 
                    action_result = self.agent.navigate(object_name)
                elif action == "pickup": 
                    action_result = self.agent.pick_up(object_name)
                elif action == "put in": 
                    action_result = self.agent.put_in(object_name)
                elif action == "toggle": 
                    action_result = self.agent.toggle(object_name)
                elif action == "open": 
                    action_result = self.agent.open(object_name)
                elif action == "close": 
                    action_result = self.agent.close(object_name)
                else:
                    print(f"Unknown action: '{action}'. Defaulting to 'observe'.")
                    action_result = self.agent.observe()
            else:
                print(f"Action '{action}' requires an object, but none was provided. Defaulting to 'observe'.")
                action_result = self.agent.observe()
            
            # check if the task is successful
            image_fp, legal_navigations, legal_interactions = action_result
            reward, success, feedback = self.round_reward(
                objectId=objectId,
                decisionmaking=decision_making,
            )
            
            self.update()
            info["success"] = info["success"] | success
            info["step"] = step
            
            if info["success"] or self.plan_end: # stop control
                self.agent.controller.stop()
                break
        
        return info
    
    def split_decision(self, decision_making):
        """
        Split the decision making string into action and object.
        """
        decision_making = decision_making.strip()
        if decision_making.startswith("end"):
            return "end", None
        elif decision_making.startswith("observe"):
            return "observe", None
        elif decision_making.startswith("move forward"):
            return "move forward", None
        elif decision_making.startswith("navigate to"):
            object_name = decision_making[len("navigate to "):].strip()
            return "navigate to", object_name
        elif decision_making.startswith("pickup"):
            object_name = decision_making[len("pickup "):].strip()
            return "pickup", object_name
        elif decision_making.startswith("put in"):
            object_name = decision_making[len("put in "):].strip()
            return "put in", object_name
        elif decision_making.startswith("toggle"):
            object_name = decision_making[len("toggle "):].strip()
            return "toggle", object_name
        elif decision_making.startswith("open"):
            object_name = decision_making[len("open "):].strip()
            return "open", object_name
        elif decision_making.startswith("close"):
            object_name = decision_making[len("close "):].strip()
            return "close", object_name
        else:
            print(f"Unknown decision making: '{decision_making}'. Defaulting to 'observe'.")
            return "observe", None
        
    
    ### round_reward function related methods ###
    
    def get_volume_distance_rate(metadata): # refer to data_engine/utils
        volumes = []
        objectid2object={}
        for obj in metadata["objects"]:
            objectid2object[obj["objectId"]]=obj
            if obj["objectType"]!="Floor":
                size=obj["axisAlignedBoundingBox"]["size"]
                v=size["x"]*size["y"]*size["z"]
                dx=obj["axisAlignedBoundingBox"]["center"]["x"]
                dz=obj["axisAlignedBoundingBox"]["center"]["z"]
                agentx=metadata["agent"]["position"]["x"]
                agentz=metadata["agent"]["position"]["z"]
                d=math.sqrt((dx-agentx)**2+(dz-agentz)**2)

                sxz = size["x"] * size["z"]
                sxy = size["x"] * size["y"]
                szy = size["y"] * size["z"]
                s = max(sxz, sxy, szy)
                if d != 0: 
                    rate = v / d
                else:
                    rate = 0 
                rate=v/d
                isnavigable=False
                if obj["visible"]==True:
                    if v<0.01:
                        isnavigable=False
                        if s>0.5 and d<10:
                            isnavigable=True
                        elif s>0.15 and d<4:
                            isnavigable=True
                        elif s>0.08 and d<2.5:
                            isnavigable=True 
                        elif v>0.005 and d<2:
                            isnavigable=True 
                        elif v>0.001 and d<1.5:
                            isnavigable=True
                        elif d<1:
                            isnavigable=True
                    else:
                        isnavigable=True
                        if rate<=0.02:
                            isnavigable=False
                            if s>0.5 and d<10:
                                isnavigable=True
                            elif s>0.15 and d<4:
                                isnavigable=True
                            elif s>0.08 and d<2.5:
                                isnavigable=True 
                            elif v>0.005 and d<2:
                                isnavigable=True 
                            elif v>0.001 and d<1.5:
                                isnavigable=True
                            elif d<1:
                                isnavigable=True                        
                volumes.append({
                    "objectId":obj["objectId"],
                    "objectType":obj["objectType"],
                    "visible":obj["visible"],
                    "volume":v,
                    "s":s,
                    "distance":d,
                    "rate":rate,
                    "isnavigable":isnavigable
                })
                sorted_volumes = sorted(volumes, key=lambda v: v["rate"])

        # save_data_to_json(sorted_volumes,"./test/navigable_list.json")
        return sorted_volumes
    
    def update_navigable_list_vtime(self): # refer to o1StyleGenerate
        self.metadata=self.agent.controller.last_event.metadata
        list = self.get_volume_distance_rate(self.metadata)
        for item in list:
            if item["isnavigable"]:  
                found = False
                for last_item in self.navigable_list:
                    if last_item["objectId"] == item["objectId"]:
                        last_item["visibleTimes"] += 1
                        found = True
                        break
                    
                if not found:
                    new_item = {
                        "objectType": item["objectType"],
                        "objectId": item["objectId"],
                        "visibleTimes": 1,
                        "choseTimes": 0
                    }
                    self.navigable_list.append(new_item)
                                
        # print("update",self.navigable_list)
        # path=f"nvrecord/{self.origin_path}/update_navigable_list.json"
        # save_data_to_json(self.navigable_list,path)
        return self.navigable_list
    
    def update(self):
        self.event = self.agent.controller.last_event
        self.metadata = self.event.metadata
        self.navigable_list=self.update_navigable_list_vtime()
    
    def maybe_find(self,objectId): # refer to o1StyleGenerate
        self.update()
        for obj in self.metadata["objects"]:
            if obj["objectId"]==objectId and obj["visible"]==True:
                agentx=self.metadata["agent"]["position"]["x"]
                agentz=self.metadata["agent"]["position"]["z"]
                objectx=obj["axisAlignedBoundingBox"]["center"]["x"]
                objectz=obj["axisAlignedBoundingBox"]["center"]["z"]
                distance = math.sqrt((objectx - agentx) ** 2 + (objectz - agentz) ** 2)
                if distance<0.8:
                    return True
        return False
    
    def is_same_objectType_show(self,objectId,find_objectId):
        self.update()
        for obj in self.metadata["objects"]:
            if obj["objectId"]==objectId:
                if obj["receptacleObjectIds"] is not None:
                    for repobjectId in obj["receptacleObjectIds"]:
                        if repobjectId.split("|")[0]==find_objectId.split("|")[0]:
                            return True
        return False
    
    def round_reward(self, objectId, decisionmaking): # refer to o1StyleGenerate
        success=False
        feedback=""
        if self.task['tasktype']=="single_search":
            if self.reward==0:
                if objectId==self.task["actions"][0]["objectId"] or objectId==self.task["actions"][0]["relatedObject"][-1] or self.is_same_objectType_show(objectId,self.task["actions"][0]["relatedObject"][-1]):
                    self.wrong_time=0
                    self.reward=self.reward+1
                    
                    feedback="The target object seems to have been successfully found."

                    self.current_action=copy.deepcopy(self.task["actions"][1])
                    self.next_action=""
                    
                else:
                    print("****** current find object",self.task["actions"][0]["relatedObject"][-1])
                    may_find_flag=self.maybe_find(self.task["actions"][0]["relatedObject"][-1])
                    if may_find_flag==True:
                        pass
                        # self.generate_o1style_data["flag"]="maybe found"
                            
                    self.wrong_time+=1
                    feedback="The target object seems not to have been found successfully."
            elif self.reward==1:
                feedback="The task seems to have been successfully completed."
                if decisionmaking=="end" or decisionmaking=="End":
                    self.reward=self.reward+1
                    success=True
                    return self.reward,success,feedback
        
        elif self.task['tasktype']=="single_pickup":
            if self.reward==0:
                if objectId==self.task["actions"][0]["objectId"] or objectId==self.task["actions"][0]["relatedObject"][-1]:
                    self.wrong_time=0
                    self.reward=self.reward+1
                     
                    feedback="The target object seems to have been successfully found, you can pick up the target object."

                    self.current_action=copy.deepcopy(self.task["actions"][1])
                    self.next_action=copy.deepcopy(self.task["actions"][2])
                else:
                    self.wrong_time+=1
                    feedback="The target object seems not to have been found successfully."
                    
            elif self.reward==1:
                if "pickup" in decisionmaking:
                    if self.current_action["objectType"] in decisionmaking:
                        self.wrong_time=0
                        self.reward=self.reward+1
                        
                        feedback="The target object seems to have been successfully picked up, and the task seems to have been completed successfully."
                        
                        self.current_action=copy.deepcopy(self.task["actions"][2])
                        self.next_action=""
                        
                    else:
                        self.wrong_time+=1
                        print("****pickup wrong object****")
                        feedback="It seems that the wrong object was picked up."
                else:
                    self.wrong_time+=1
                    feedback="The target object seems not to have been successfully picked up."  
            
        elif self.task['tasktype']=="single_search_from_closerep":
            if self.reward==0:
                if objectId==self.task["actions"][0]["objectId"] or objectId==self.task["actions"][0]["relatedObject"][-1]:
                    self.wrong_time=0
                    self.reward=self.reward+1
                    feedback="The target container seems to have been successfully located. You can now open it to check if the target object is inside."

                    self.current_action=copy.deepcopy(self.task["actions"][1])
                    self.next_action=copy.deepcopy(self.task["actions"][2])

                    self.plan_objects_list=[]
                else:
                    self.wrong_time+=1
                    feedback="The target object seems not to have been found successfully."
             
            elif self.reward==1:
                if "open" in decisionmaking:
                    if self.current_action["objectType"] in decisionmaking:
                        self.wrong_time=0
                        self.reward=self.reward+1
                        
                        feedback="The target container seems to have been successfully opened, and the task seems to have been completed successfully."                        
                        self.current_action=copy.deepcopy(self.task["actions"][2])
                        self.next_action=""
                        
                            
                    else:
                        self.wrong_time+=1
                        print("****open wrong object****")
                        feedback="It seems that the wrong container was chosen to be opened."
            
            
            return self.reward,success,feedback
        
        elif self.task['tasktype']=="single_pickup_from_closerep":
            if self.reward==0:
                if objectId==self.task["actions"][0]["objectId"] or objectId==self.task["actions"][0]["relatedObject"][-1]:
                    self.wrong_time=0
                    self.reward=self.reward+1
                    feedback="The target container seems to have been successfully located. You can now open it to check if the target object is inside."

                    self.current_action=copy.deepcopy(self.task["actions"][1])
                    self.next_action=copy.deepcopy(self.task["actions"][2])

                    self.plan_objects_list=[]
                else:
                    self.wrong_time+=1
                    feedback="The target object seems not to have been found successfully."
             
            elif self.reward==1:
                if "open" in decisionmaking:
                    if self.current_action["objectType"] in decisionmaking:
                        self.wrong_time=0
                        self.reward=self.reward+1
                        
                        feedback="The target container seems to have been successfully opened. You can now pick up the object inside."                        
                        self.current_action=copy.deepcopy(self.task["actions"][2])
                        self.next_action=copy.deepcopy(self.task["actions"][3])
                        
                            
                    else:
                        self.wrong_time+=1
                        feedback="It seems that the wrong container was chosen to be opened."
          
            elif self.reward==2:
                if "pickup" in decisionmaking:
                    if self.current_action["objectType"] in decisionmaking:
                        self.wrong_time=0
                        self.reward=self.reward+1

                        feedback="The target object has been successfully picked up. You can now close the container."
                        
                        self.current_action=copy.deepcopy(self.task["actions"][3])
                        self.next_action=copy.deepcopy(self.task["actions"][4])
                        
                    else:
                        self.wrong_time+=1
                        print("****pickup wrong object****")
                        feedback="It seems that the wrong object was picked up."
                else:
                    self.wrong_time+=1
                    feedback="The target object seems not to have been successfully picked up." 
                    
            elif self.reward==3:
                if "close" in decisionmaking:
                    if self.current_action["objectType"] in decisionmaking:
                        self.wrong_time=0
                        self.reward=self.reward+1
                        
                        feedback="The target container seems to have been successfully closed, and the task seems to have been completed successfully."
                        self.current_action=copy.deepcopy(self.task["actions"][4])
                        self.next_action=""
                        
                            
                    else:
                        self.wrong_time+=1
                        print("****open wrong object****")
                        feedback="It seems that the wrong container was chosen to be closed."                 
        
        elif self.task['tasktype']=="single_toggle":
            if self.reward==0:
                if objectId==self.task["actions"][0]["objectId"] or objectId==self.task["actions"][0]["relatedObject"][-1]:
                    self.wrong_time=0
                    self.reward=self.reward+1 
                    feedback="The target object appears to have been successfully located. You can now toggle the target object."

                    self.current_action=copy.deepcopy(self.task["actions"][1])
                    self.next_action=copy.deepcopy(self.task["actions"][2])
                else:
                    self.wrong_time+=1
                    feedback="The target object seems not to have been found successfully."
                    
            elif self.reward==1:
                if "toggle" in decisionmaking:
                    if self.current_action["objectType"] in decisionmaking:
                        self.wrong_time=0
                        self.reward=self.reward+1

                        feedback="The target object appears to have been toggled successfully, and the task seems to have been completed successfully."
                                                
                        self.current_action=copy.deepcopy(self.task["actions"][2])
                        self.next_action=""
                        
                    else:
                        self.wrong_time+=1
                        print("****pickup wrong object****")
                        feedback="It seems that the wrong object was picked up."
                else:
                    self.wrong_time+=1
                    feedback="The target object seems not to have been successfully picked up."                    
                
            
            return self.reward,success,feedback
        
        elif self.task['tasktype']=="pickup_and_put":
            if self.reward==0:
                if objectId==self.task["actions"][0]["objectId"] or objectId==self.task["actions"][0]["relatedObject"][-1]:
                    self.wrong_time=0
                    self.reward=self.reward+1
                    feedback="The target object seems to have been successfully found, you can pick up the target object."
                    
                    self.current_action=copy.deepcopy(self.task["actions"][1])
                    self.next_action=copy.deepcopy(self.task["actions"][2])
                else:
                    self.wrong_time+=1
                    feedback="The target object seems not to have been found successfully."
            elif self.reward==1:
                if "pickup" in decisionmaking:
                    if self.current_action["objectType"] in decisionmaking:
                        self.wrong_time=0
                        self.reward=self.reward+1
                        
                        feedback="The target object seems to have been successfully picked up."
                        
                        self.current_action=copy.deepcopy(self.task["actions"][2])
                        self.next_action=copy.deepcopy(self.task["actions"][3])
                        
                    else:
                        self.wrong_time+=1
                        print("****pickup wrong object****")
                        feedback="It seems that the wrong object was picked up."
                else:
                    self.wrong_time+=1
                    feedback="The target object seems not to have been successfully picked up."                    
                        
            elif self.reward==2:
                if objectId==self.current_action["objectId"] or objectId==self.current_action["relatedObject"][-1]:
                    self.wrong_time=0
                    self.reward=self.reward+1

                    feedback="The container seems to have been successfully found."
                    
                    self.current_action=copy.deepcopy(self.task["actions"][3])
                    self.next_action=copy.deepcopy(self.task["actions"][4])
                    
                else:
                    self.wrong_time+=1
                    feedback="The container seems not to have been found successfully."
                
                
            elif self.reward==3:
                if "put" in decisionmaking:
                    if self.current_action["objectType"] in decisionmaking:
                        self.wrong_time=0
                        self.reward=self.reward+1
                        
                        if self.metadata["errorMessage"]=="":
                            # 更新 action 和 feedback 
                            feedback="The object in hand seems to have been successfully placed."# place in/on 会对最终的 thinking产生影响
                            self.current_action=copy.deepcopy(self.task["actions"][4]) # 最后一步end
                            self.next_action=""
                        
                        else:
                            self.current_action=copy.deepcopy(self.task["actions"][4]) # 最后一步end
                            self.next_action=""
                            
                    else:
                        self.wrong_time+=1
                        print("****put wrong object****")
                        feedback="It seems that the wrong container was chosen for placement."
                else:
                    self.wrong_time+=1
                    feedback="The placement location was found before, but it seems that placement was not chosen."
            
            
            
            return self.reward,success,feedback
        
        elif self.task['tasktype']=="pickup_and_put_in_closerep":
            if self.reward==0:
                if objectId==self.task["actions"][0]["objectId"] or objectId==self.task["actions"][0]["relatedObject"][-1]:
                    self.wrong_time=0
                    self.reward=self.reward+1
                    feedback="The target object seems to have been successfully found, you can pick up the target object."
                    self.current_action=copy.deepcopy(self.task["actions"][1])
                    self.next_action=copy.deepcopy(self.task["actions"][2])
                    self.plan_objects_list=[]
                else:
                    self.wrong_time+=1
                    feedback="The target object seems not to have been found successfully."
                    
            elif self.reward==1:
                if "pickup" in decisionmaking:
                    if self.current_action["objectType"] in decisionmaking:
                        self.wrong_time=0
                        self.reward=self.reward+1

                        feedback="The target object has been successfully picked up. You may now proceed to the next location to place it."
                        
                        self.current_action=copy.deepcopy(self.task["actions"][2])
                        self.next_action=copy.deepcopy(self.task["actions"][3])
                        
                    else:
                        self.wrong_time+=1
                        print("****pickup wrong object****")
                        feedback="It seems that the wrong object was picked up."
                else:
                    self.wrong_time+=1
                    feedback="The target object seems not to have been successfully picked up."                    
                        
            elif self.reward==2:
                if objectId==self.current_action["objectId"] or objectId==self.current_action["relatedObject"][-1]:
                    self.wrong_time=0
                    self.reward=self.reward+1

                    feedback="The container seems to have been successfully located. You are now holding the target object and can proceed to place it in the container, but you need to open the container first."                    
                    self.current_action=copy.deepcopy(self.task["actions"][3])
                    self.next_action=copy.deepcopy(self.task["actions"][4])
                    
                    self.plan_objects_list=[]
                    
                else:
                    self.wrong_time+=1
                    feedback="The container seems not to have been found successfully."
                    
            elif self.reward==3:
                if "open" in decisionmaking:
                    if self.current_action["objectType"] in decisionmaking:
                        self.wrong_time=0
                        self.reward=self.reward+1
                        
                        feedback="The target container seems to have been successfully opened. You can now place the object you are holding inside."                        
                        self.current_action=copy.deepcopy(self.task["actions"][4])
                        self.next_action=copy.deepcopy(self.task["actions"][5])
                        
                            
                    else:
                        self.wrong_time+=1
                        print("****open wrong object****")
                        feedback="It seems that the wrong container was chosen to be opened."
    
                
            elif self.reward==4:
                if "put" in decisionmaking:
                    if self.current_action["objectType"] in decisionmaking:
                        self.wrong_time=0
                        self.reward=self.reward+1
                        
                        feedback="The object in hand seems to have been successfully placed, and the task seems to have been successfully completed."
                        self.current_action=copy.deepcopy(self.task["actions"][5])
                        self.next_action=""                       
                            
                    else:
                        self.wrong_time+=1
                        print("****put wrong object****")
                        feedback="It seems that the wrong container was chosen for placement."
                else:
                    self.wrong_time+=1
                    feedback="The placement location was found before, but it seems that placement was not chosen."
            
            
            return self.reward,success,feedback            
            
        elif self.task['tasktype']=="pickup_from_closerep_and_put":
            if self.reward==0:
                if objectId==self.task["actions"][0]["objectId"] or objectId==self.task["actions"][0]["relatedObject"][-1]:
                    self.wrong_time=0
                    self.reward=self.reward+1
                    feedback="The target container seems to have been successfully located. You can now open it to check if the target object is inside."
                    self.current_action=copy.deepcopy(self.task["actions"][1])
                    self.next_action=copy.deepcopy(self.task["actions"][2])
                    self.plan_objects_list=[]
                else:
                    self.wrong_time+=1
                    feedback="The target object seems not to have been found successfully."
             
            elif self.reward==1:
                if "open" in decisionmaking:
                    if self.current_action["objectType"] in decisionmaking:
                        self.wrong_time=0
                        self.reward=self.reward+1
                        
                        feedback="The target container seems to have been successfully opened. You can now pick up the object inside."                        
                        self.current_action=copy.deepcopy(self.task["actions"][2])
                        self.next_action=copy.deepcopy(self.task["actions"][3])    
                    else:
                        self.wrong_time+=1
                        print("****open wrong object****")
                        feedback="It seems that the wrong container was chosen to be opened."
          
            elif self.reward==2:
                if "pickup" in decisionmaking:
                    if self.current_action["objectType"] in decisionmaking:
                        self.wrong_time=0
                        self.reward=self.reward+1
                        feedback="The target object has been successfully picked up. You can now close the container and proceed to the next location to place the object."
                        
                        self.current_action=copy.deepcopy(self.task["actions"][3])
                        self.next_action=copy.deepcopy(self.task["actions"][4])
                        
                    else:
                        self.wrong_time+=1
                        print("****pickup wrong object****")
                        feedback="It seems that the wrong object was picked up."
                else:
                    self.wrong_time+=1
                    feedback="The target object seems not to have been successfully picked up." 
                    
            elif self.reward==3:
                if "close" in decisionmaking:
                    if self.current_action["objectType"] in decisionmaking:
                        self.wrong_time=0
                        self.reward=self.reward+1
                        
                        feedback="The target container seems to have been successfully closed. You can now proceed to the next location to place the object."
                        self.current_action=copy.deepcopy(self.task["actions"][4])
                        self.next_action=copy.deepcopy(self.task["actions"][5])
                        
                            
                    else:
                        self.wrong_time+=1
                        print("****open wrong object****")
                        feedback="It seems that the wrong container was chosen to be closed."                   
                        
            elif self.reward==4:
                if objectId==self.current_action["objectId"] or objectId==self.current_action["relatedObject"][-1]:
                    self.wrong_time=0
                    self.reward=self.reward+1
                    feedback="The container seems to have been successfully located. You are now holding the target object and can proceed to place it inside the container."
                    
                    self.current_action=copy.deepcopy(self.task["actions"][5])
                    self.next_action=copy.deepcopy(self.task["actions"][6])
                    self.plan_objects_list=[]
                    
                else:
                    self.wrong_time+=1
                    feedback="The container seems not to have been found successfully."
                        
            elif self.reward==5:
                if "put" in decisionmaking:
                    if self.current_action["objectType"] in decisionmaking:
                        self.wrong_time=0
                        self.reward=self.reward+1
                        
                        feedback="The object in hand seems to have been successfully placed, and the task seems to have been successfully completed."
                        self.current_action=copy.deepcopy(self.task["actions"][6])
                        self.next_action=""                       
                            
                    else:
                        self.wrong_time+=1
                        print("****put wrong object****")
                        feedback="It seems that the wrong container was chosen for placement."
                else:
                    self.wrong_time+=1
                    feedback="The placement location was found before, but it seems that placement was not chosen."
           
            return self.reward,success,feedback 
           
        elif self.task['tasktype']=="pickup_from_closerep_and_put_in_closerep":
            if self.reward==0:
                if objectId==self.task["actions"][0]["objectId"] or objectId==self.task["actions"][0]["relatedObject"][-1]:
                    self.wrong_time=0
                    self.reward=self.reward+1
                    feedback="The target container seems to have been successfully located. You can now open it to check if the target object is inside."
                    
                    self.current_action=copy.deepcopy(self.task["actions"][1])
                    self.next_action=copy.deepcopy(self.task["actions"][2])
                    
                    self.plan_objects_list=[]
                else:
                    self.wrong_time+=1
                    feedback="The target object seems not to have been found successfully."
             
            elif self.reward==1:
                if "open" in decisionmaking:
                    if self.current_action["objectType"] in decisionmaking:
                        self.wrong_time=0
                        self.reward=self.reward+1
                        
                        feedback="The target container seems to have been successfully opened. You can now pick up the object inside."                        
                        self.current_action=copy.deepcopy(self.task["actions"][2])
                        self.next_action=copy.deepcopy(self.task["actions"][3])
                        
                            
                    else:
                        self.wrong_time+=1
                        print("****open wrong object****")
                        feedback="It seems that the wrong container was chosen to be opened."
          
            elif self.reward==2:
                if "pickup" in decisionmaking:
                    if self.current_action["objectType"] in decisionmaking:
                        self.wrong_time=0
                        self.reward=self.reward+1
                        
                        feedback="The target object has been successfully picked up. You can now close the container and proceed to the next location to place the object."
                        
                        self.current_action=copy.deepcopy(self.task["actions"][3])
                        self.next_action=copy.deepcopy(self.task["actions"][4])
                        
                    else:
                        self.wrong_time+=1
                        print("****pickup wrong object****")
                        feedback="It seems that the wrong object was picked up."
                else:
                    self.wrong_time+=1
                    feedback="The target object seems not to have been successfully picked up." 
                    
            elif self.reward==3:
                if "close" in decisionmaking:
                    if self.current_action["objectType"] in decisionmaking:
                        self.wrong_time=0
                        self.reward=self.reward+1
                        
                        feedback="The target container seems to have been successfully closed. You can now proceed to the next location to place the object."
                        self.current_action=copy.deepcopy(self.task["actions"][4])
                        self.next_action=copy.deepcopy(self.task["actions"][5])
                        
                            
                    else:
                        self.wrong_time+=1
                        print("****open wrong object****")
                        feedback="It seems that the wrong container was chosen to be closed."                   
                        
            elif self.reward==4:
                if objectId==self.current_action["objectId"] or objectId==self.current_action["relatedObject"][-1]:
                    self.wrong_time=0
                    self.reward=self.reward+1
                    
                    feedback="The target container seems to have been successfully located. You can now open it and place the object you are holding inside."
                    self.current_action=copy.deepcopy(self.task["actions"][5])
                    self.next_action=copy.deepcopy(self.task["actions"][6])

                    self.plan_objects_list=[]
                    
                else:
                    self.wrong_time+=1
                    feedback="The container seems not to have been found successfully."

            elif self.reward==5:
                if "open" in decisionmaking:
                    if self.current_action["objectType"] in decisionmaking:
                        self.wrong_time=0
                        self.reward=self.reward+1
                        
                        feedback="The target container seems to have been successfully opened. You can now place the object you are holding inside."                        
                        self.current_action=copy.deepcopy(self.task["actions"][6])
                        self.next_action=copy.deepcopy(self.task["actions"][7])    
                    else:
                        self.wrong_time+=1
                        print("****open wrong object****")
                        feedback="It seems that the wrong container was chosen to be opened."
    
            elif self.reward==6:
                if "put" in decisionmaking:
                    if self.current_action["objectType"] in decisionmaking:
                        self.wrong_time=0
                        self.reward=self.reward+1
                        
                        feedback="The object in hand seems to have been successfully placed, and the task seems to have been successfully completed."
                        self.current_action=copy.deepcopy(self.task["actions"][7])
                        self.next_action=""                              
                    else:
                        self.wrong_time+=1
                        print("****put wrong object****")
                        feedback="It seems that the wrong container was chosen for placement."
                else:
                    self.wrong_time+=1
                    feedback="The placement location was found before, but it seems that placement was not chosen."
           
            return self.reward,success,feedback 
            

        return self.reward,success,feedback
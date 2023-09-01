# I have spent some years working on factories, building factory stock management systems and analysing SQL data to simulate a factory OS system for production planning
# I always used to use excel vba to ping SQL queries off of SQL databases and Access databases, this document is written in python a language which I will be learning over the next 3 months
# for now this is just pure python, with a few comments to highlight further iterations that I wish to make to the code.

# when run you will follow the adventures of Kevin as he tries to operate machine - MC001. Kevin is my father-in-laws name.
# The machine will normally produce an SQL update string every cycle to represent part throughput, but it will occasioanlly break down
# Kevin will try to keep up with maintenance and repairs, but ultimately he will fail, unless he can learn more skills quickly!
# Kevin gains more skills by successfully completing repairs to a very high standard, however if he fumbles difficulties rise and so do breakdown occurences.

# I started off figuring out how to string functions together, but found that many values I wanted to hold sacred were getting changed by the code.
# This iteration has focused on dictionary definition, being careful where I link dictionaries directly and where I make copies and learning the intricacies of which methods can be used on which variable types
# I decided to make it a little fairer on Kevin and added a high score system - There is no skill here its a random chance.

# I start my Data_science course on the 12/9/2023 and in the 11 days before then, 
# I hope to break this code down and add threads to it to have multiple operators working on multiple machines at one time
# I also have a spare hard drive which I will be creating an SQL server on to practice my database set up, reference and updating skills.
# after I start my course I look forward to seeing what else I could add to this project!

# to play with this code, unhash the machine_temp_vars tables to see different start up conditions and change kevins abilities in the operators table.

import time 
import random
import numpy as np
import threading
import sys

# Shared Dictionary creation
machine_temp_vars = {}
part_vars = {}
machine_attributes = {}
part_attributes = {}
operators ={}
order_details = {}

# locks for access to shared data
machine_vars_lock = threading.Lock()
part_vars_lock = threading.Lock()
attributes_lock = threading.Lock()
operator_lock = threading.Lock()

#---------------
# Operator Stats
#---------------

# op_name -------- Operators name for visual display of their actions
# op_react_time -- Time it will take for the operator to react to problems
# op_judge_time -- Time it takes for operator to analyse and assess problems or conditions
# op_skill_low --- Lower threshold of skill. Lower numbers are less likely to cause delays when fixing issues. A rookie or a professional might have low thresholds, but an overconfident amateur will have a high one.
# op_skill_high -- Upper threshold of skill. Higher numbers are able to solve a wider array of problems
# op_set_multi --- How long it takes operators to perform tasks compared to the nominal value.

with operator_lock:
    operators["OP01"] = {
        "op_name": "Kevin",
        "op_react_time": 15,
        "op_judge_time": 2,
        "op_skill_low": 20,
        "op_skill_high": 60,
        "op_set_multi": 2,
        "tenacity": 5
    }

#--------------
# MACHINE STATS
#--------------

# ---- THESE ITEMS WILL BE STORED IN SQL ONCE I AM HAPPY WITH THE FUNCTIONALITY ----

# machine_id ------- unique id for each machine defines attribute dictionary and temporary variables dictionaries.
# start_up_time ---- Time the machine takes to get started, e.g getting up to speed or settling vibrations.
# set_up_time ------ nominal value for how long it takes to set the machine, used in initial set up and repairs and modified by operator speed multiplier
# set_part --------- Part value that the machine is currently set to. This is compared to part_type being run on the machine to see if set up can be skipped.
# cycle_t ---------- Placeholder machine cycle time for development. Later this will become more complex as we include machine rail speeds and tool quality vs part materials being cut
# run_time --------- Placeholder value to simulate machine operation, later this will become related to shift length or number of parts in teh batch from the order.
# breakdown_hz ----- Frequency of issues, this will later be paired with things like machine age or Manufacturer quality to represent decay and quality of investment
# init_tool_health - Placeholder value to represent how sturdy Machine tools are. This will be moved out of machines once we develop a tooling dictionary.
# part_id ---------- Unique to each part in this machine for tracking, irrespective of Part type.

#----------------
# BREAKDOWN STATS
#----------------

# ---- THESE ITEMS WILL FORM THE BREAKDOWN DICTIONARY, NESTED WITHIN THE MACHINES ATTRIBUTES DICT. ALSO TO BE STORED IN SQL ----

# Load Error stats
# ----------------------- The part has misloaded entering the machine. The machine will not run while this error is present.
# le_chance ------------- As the first type of error, le_chance is always 100 forming the baseline of our percentage chance error type.
# le_fix_time ----------- Defines the nominal length of time for an operator to fix the issue, modified by operator speed multiplyer
# le_repair_difficulty -- This defines how hard the load error is to fix, on simple machines this will be a low number, but if machines have complex loading systems or robotic loaders this could go up

# Tool Break Stats
# ------------------------ The insert in the machine snaps and requires replacement. The machine will keep running while this error is present, but running in this state could lead to bigger problems.
# t_b_chance ------------- As the second error type, the difference between le_chance and t_b_chance will determine how often Load Errors occur. This number can be changed by the machine as it gets older or as it has larger breakdowns or crashes and when a tool breaks a larger breakdown is more likely
# t_b_effect ------------- Random number whose limits determine how bad the breakage is. If 100 then the machines output will either do nothing too the parts or create extremely variable results.
# t_b_fix_time ----------- Determines the nominal length of time it takes for an operator to repair the broken tool. Modified by operator speed.
# t_b_repair_difficulty -- This defines how hard it is to repair the tool breakage.

# Setting drift Stats
# ------------------------ This error is difficult to detect and difficult to fix, though the repair does not take long for an operator with a high skill. Detection difficulty represents lack of spc on the machine, which will be further explored later.
# s_d_chance ------------- As the last error type, the distance between this value and t_b_chance will determine how often tool breakages occur and the size of this number will define how often settings may drift.
# s_d_effect ------------- A random number with positive and negative limits, representing total potential percentile drift. The random fluctuations up and down make it even harder for operators to detect.
# s_d_fix_time ----------- Length of time it takes for the operator to fix the setting drift, once detected.
# s_d_repair_difficulty -- The difficulty of the setting drift repair. This high value may be used to make a competent, but not skilled operator believe they have fixed the error when they have not.



machines = ["MC001","MC002"] # This list simulates the machines currently available to us.

with machine_vars_lock:
    for machine_id in machines: # cycle through machines
        if machine_id not in machine_temp_vars: # Check if machine ID exists in the shared dictionary
            machine_temp_vars[machine_id] = {}

#------------------
# MACHINE_TEMP_VARS
#------------------

# These are provided for reference only, to visualise how the machine_temp_vars dictionaries should look, however these dictionaries are most often wiped and created by the machine as it runs, so they are essentially semi-permanent.
#
#machine_temp_vars["MC001"] = { # define dictionary by machine_id
#     "machine_state": False, #defines whether the machine is currently running
#     "init_params": { # init params are for setting up the machine, they are copies of the drawing refs from the part attributes table. They should be copied if used to implement changes to preserve their values.
#         "DRW001": { # defines different settings for different part types on the machine.
#             "diameter01": {"mean": 20, "std_dev": 0.1},
#             "length01": {"mean": 38, "std_dev": 0.2},
#             "diameter02": {"mean": 18, "std_dev": 0.2},
#             "length02": {"mean": 38, "std_dev": 0.4} # example entry, these mean and std_dev values are copied from the part_attributes table
#             }
#         },
#     "variables_param": { # variables param are taken from two places, a copy of the init_params and also from the machine attributes table. In generaly means are set by the init_params and the std_dev by the machine attributes, but future functions will be added to represent machine age and wear seperate from tool wear. (e.g. backlash in the rack and pinion)
#         "DRW001": { # defines different settings for different part types on the machine.
#             "diameter01": {"mean": 20, "std_dev": 0.06},
#             "length01": {"mean": 38, "std_dev": 0.1},
#             "diameter02": {"mean": 18, "std_dev": 0.06},
#             "length02": {"mean": 38, "std_dev": 0.1} # example entry.
#             }
#         },
#     "running_vars": { # the running_vars are the variables that contribute to the actual parts produced. they are initially copies of the variables_param, but regularly changed by machine operation.
#         "set_part": "DRW001",
#         "part_id": {"DRW001":247},
#         "DRW001": { # defines different settings for different part types on the machine.
#             "diameter01": {"mean": 20, "std_dev": 0.06},
#             "length01": {"mean": 38, "std_dev": 0.1},
#             "diameter02": {"mean": 18, "std_dev": 0.06},
#             "length02": {"mean": 38, "std_dev": 0.1} # example entry.
#             }
#         },
#     "tooling_param": { #stores parameters that are used during the running of the machine
#         "tool_health": 60, # defines the health of the current working tool
#         "tool_nom_dev": {
#             "diameter01": { "std_dev": 0.06},
#             "length01": { "std_dev": 0.1},
#             "diameter02": { "std_dev": 0.06},
#             "length02": { "std_dev": 0.1} # defines the nominal standard deviation to define mean drift over time
#             }
#         }
# }

# The dictionaries below will be assimilated into a for loop similar to the temp_vars above once machine data is stored in SQL
# for machine_id in machines:
#   if machine_id not in machine_attributes:
#       machine_attributes[machine_id] = {}
#   machine_attributes[machine_id]["start_up_time"] = SQL query
#   etc. Will work on this and best practices when I have my SQL server set up

with attributes_lock:
    machine_attributes["MC001"] = {
        "start_up_time" : 2,
        "set_up_time" : 2,
        "cycle_t": 1,
        "run_time": 120,
        "breakdown_hz": 20,
        "breakdown_table": {
            "l_e": {
                "chance": 100,
                "machine_state": False,
                "fix_time": 5,
                "repair_difficulty": 5
                },
            "t_b": {
                "chance": 25,
                "machine_state": True,
                "fix_time": 5,
                "repair_difficulty": 50
                },
            "s_d": {
                "chance": 10,
                "machine_state": True,
                "fix_time": 4,
                "repair_difficulty": 70
                }
            },
        "select_tool": {
            "standard" : {
                "init_tool_health": 100,
                "devs": {
                    "diameter": {"std_dev": 0.05},
                    "length": {"std_dev": 0.01}
                    #"perpendicularity": {"std_dev": 0.005},
                    #"cylindricity": {"std_dev": 0.002}
                    #"concentricity": {"std_dev": 0.001}, 
                    #"surface finish": {"std_dev": 0.1},
                    #"roundness": {"std_dev": 0.005}
                    }
                },
                "performance" : {
                "init_tool_health": 100,
                "devs": {
                    "diameter": {"std_dev": 0.005},
                    "length": {"std_dev": 0.001}
                    #"perpendicularity": {"std_dev": 0.0005},
                    #"cylindricity": {"std_dev": 0.0002}
                    #"concentricity": {"std_dev": 0.0001}, 
                    #"surface finish": {"std_dev": 0.001},
                    #"roundness": {"std_dev": 0.005}
                    }
                }
            }
    }

    machine_attributes["MC002"] = {
        "start_up_time" : 10,
        "set_up_time" : 20,
        "set_part": "DRW001",
        "cycle_t": 2,
        "run_time": 500,
        "breakdown_hz": 10,
        "breakdown_table": {
            "l_e":{
                "chance": 100,
                "machine_state": False,
                "effect": 0,
                "fix_time": 2,
                "repair_difficulty": 5
                },
            "t_b": {
                "chance": 40,
                "machine_state": False,
                "effect": random.randint(10,100), # may replace this with upper and lower limits later and move the random number generation into the breakdown function
                "fix_time": 30,
                "repair_difficulty": 70
                },
            "s_d": {
                "chance": 20,
                "machine_state": True,
                "effect": random.randint(-10,10), # same here
                "fix_time": 4,
                "repair_difficulty": 70
                },
            },
        "select tool": {
            "standard" : {
                "init_tool_health": 100,
                "diameter": {"std_dev": 0.05},
                "length": {"std_dev": 0.01}
                #"perpendicularity": {"std_dev": 0.005},
                #"cylindricity": {"std_dev": 0.002}
                #"concentricity": {"std_dev": 0.001}, 
                #"surface finish": {"std_dev": 0.1},
                #"roundness": {"std_dev": 0.005}
                },
                "performance" : {
                "init_tool_health": 100,
                "diameter": {"std_dev": 0.005},
                "length": {"std_dev": 0.001}
                #"perpendicularity": {"std_dev": 0.0005},
                #"cylindricity": {"std_dev": 0.0002}
                #"concentricity": {"std_dev": 0.0001}, 
                #"surface finish": {"std_dev": 0.001},
                #"roundness": {"std_dev": 0.005}
                }
            }
    }

#-----------
# PART STATS
#-----------

# part_type ---- Part Drawing Number unique to each part, used to reference the part_attributes shared dictionary and also for SQL queries.
# total ops ---- number of operations the part requires for completion, a drawing of a bar with a chamfer would require two operations, operations relate to individual types of dimension the machine is capable of producing. In the example used diameter1 would refer to the rod diameter, diamater 2 would relate to the final chamfer diameter. etc.

with attributes_lock:
    part_attributes["DRW001"] = {
        "total_ops": 2,
        "target_dims": {
            "diameter01": {"mean": 20, "std_dev": 0.1},
            "length01": {"mean": 38, "std_dev": 0.2},
            "diameter02": {"mean": 18, "std_dev": 0.2},
            "length02": {"mean": 38, "std_dev": 0.4}
        }
    }
    part_attributes["DRW002"] = {
        "total_ops": 3,
        "target_dims": {
            "diameter01": {"mean": 25, "std_dev": 0.1},
            "length01": {"mean": 30, "std_dev": 0.2},
            "diameter02": {"mean": 20, "std_dev": 0.2},
            "length02": {"mean": 40, "std_dev": 0.4},
            "diameter03": {"mean": 20, "std_dev": 0.3},
            "length03": {"mean": 60, "std_dev": 0.6}
        }
    }

#-------------------------
# Machine Variables set up
#-------------------------

# Creating initial Parameters --- these are the masters

def create_inits(
        drawing_specs, #drawing specs drawn from part_attribute table. This function will only read and copy these, so copying is not necessary
        machine_temp_vars #feed through to preserve machine id
        ): 
    machine_temp_vars["init_params"][part_type] = {} #create or empty the init_params
    for key, value in drawing_specs.items(): #cycle drawing elements
        machine_temp_vars["init_params"][part_type][key] = value.copy() # copy drawing elements to the init_params

# creating variable parameters for the machine, these are copies of the init params to be modified by the machine

def master_params( # This function should always create variables_param dictionaries for the machine in question. It represents a part change and a tool change as both means and std_devs are affected.
        init_params, # Feed a copy of the init_params to prevent alterations to the master values
        machine_tool, # Tools dictionary is linked to the variables table so long term wear is stored in the machine. running_vars hold temporary values for inserts and do not affect this function.
        machine_temp_vars #feed through to preserve machine id
        ):
    machine_temp_vars["variables_param"][part_type] = {} #empty variables_param
    for key, dict_value in init_params.items(): #Cycle Through init param elements
        if "mean" in dict_value: # looks to see if the key has a "mean" value stored, which it should.
            machine_temp_vars["variables_param"][part_type][key] = {"mean": init_params[key]["mean"], "std_dev": 0} # create copies of mean value key on each cycle from the init_params to the copied dictionary
    for key, dict_value in machine_temp_vars["variables_param"][part_type].items():
        if "std_dev" in dict_value: #finds init_param keys for std_devs
            var_key = key[:-2] #removes the number from the key to leave the base dimension, Only suitable up to 99 operations, if a part has 100 or more then key format will need to be updated - something to consider before establishing the SQL server
            tool_value = machine_tool["devs"][var_key]["std_dev"] #links the "value stored in the std_dev tuple for the dimension"
            machine_temp_vars["variables_param"][part_type][key]["std_dev"] = tool_value # sets the temp_variables_param dictionary to hold machine wear values.

# reset the running_vars to be equal to the variables vars

def temp_param_reset( #function to reset both running variable means and distributions. Represents a machine set up that includes a tool change
        variables_param_copy, # copy required to prevent accidental changes.
        machine_temp_vars #feed through to preserve machine id
        ):
    machine_temp_vars["running_vars"][part_type] = {} #empty running_vars
    for key, value in variables_param_copy.items(): #Cycle Through created keys
        machine_temp_vars["running_vars"][part_type][key] = value.copy() #create copies of all keys and values in variables_param

#--------------------------
# MACHINE SETTING FUNCTIONS
#--------------------------

# TOOL CHANGE FUNCTION

def tool_change( # Function to simulate an operator performing a tool change on the machine. means and std-devs are affected. This currently works with 100% success rate, but failure related to operator skill could be added in future as this is a reasonably common cause of discrepancies.
        machine_id,# feed function machine_id, for GUI responses and running_vars dictionary definition, copies not necessary
        machine_values_copy, # feed the copies of machine_values which can be adjusted and disposed of. do NOT feed Master_attributes directly as this will alter the starting health of tools, leading to an infinite decay and 0 init_health which lead to zero divide errors
        machine_temp_vars, #direct link to variables dictionary
        current_op, # feed the function the operator dictionary specific to the current operator. This function does not change variables, only reads them, so copying is not necessary.
        variables_param_copy #copied at root of this function tree to prevent all functions from affecting the main variables_param dictionary
        ): 
    
    #machine variables
    set_up_time = machine_values_copy["set_up_time"]
    temp_tool_health = machine_values_copy["select_tool"][tool]["init_tool_health"] #set the tool health variable to match the initial health values
    tool_vars = machine_temp_vars["tooling_param"] #direct link necessary to make updates

    #operator variables
    op_name = current_op["op_name"]
    op_set_multi = current_op["op_set_multi"]

    #unique variables
    settime = set_up_time*op_set_multi# define the time that the tool change will take, this is equal to the setting time multiplied by the operator setting multiplier, different operators will complete this task at different speeds.

    #function begins
    if isinstance(temp_tool_health, int) and temp_tool_health > 0: #error handling to prevent zero divide errors
        print(f"{op_name} has gathered their tools and started to change the tool inserts on {machine_id}") # GUI readout to make Observer aware of what is going on.
        time.sleep(settime) # wait to simulate repair occuring
        tool_vars["tool_health"] = temp_tool_health # set the machine_temp_vars table to the value of the copied init_tool_health from machine attributes
        tool_vars["tool_nom_dev"] = {} # create/empty the tool_nom_dev dictionary
        for key, dict_value in variables_param_copy.items(): # cycle through keys in variables param
            if "std_dev" in dict_value: # find std_dev values
                tool_vars["tool_nom_dev"][key] = dict_value # set tooling variable table to hold std_devs with the same key as the variables params
        temp_param_reset( #call the function to reset the running vars to the values held in the variables table 
            variables_param_copy=variables_param_copy, # Variable is read and copied only, no need for copying on feed. (mean - copied from inits, copied from drawings. std_dev copied from machine attributes, sometimes altered by major breakdowns)
            machine_temp_vars=machine_temp_vars #feed through to preserve machine id #feed through to preserve machine id
            )
    else:
        print(f"There is a problem with the init tool health on {machine_id}. It is either 0, less than 0 or not a number.") #error handling to prevent zero divide errors

# OPEARTOR ADJUSTMENT FUNCTION

def set_means( # This function simulates a part change without changing tooling, or an operator adjusting the target value of the machine without making any other changes. This function sets the mean value to the exact drawing target, but it could be iterated on to allow more skilled operators to set the target a little to one side of the target for step profile spc patterns
        variables_param_copy, # This function only reads, compares and copies the variables params. Feeding a copy is not necessary.
        running_vars # linked to the running vars in machine_temp_vars. Changes here affect the running vars
        ): # feed the function the two sets of variables to compare
    keys_to_remove = [] #create an empty list to store keys for deletion
    for key, dict_value in running_vars.items(): #cycle through keys and values in running_vars{}. e.g. key = "diameter", dict_value = "mean"
        if key not in variables_param_copy: #Find keys in running_vars that do not exist in variables param
            keys_to_remove.append(key)  # populate key deletion list with keys that do not exist in variables_param
        elif dict_value == "mean":  # if the key does exist, Check if the value is "mean"
                running_vars[key]["mean"] = variables_param_copy[key]["mean"]  # Copy the mean key and value from variables
    
    for key in keys_to_remove: #cycle through keys marked for deletion
        del running_vars[key] #and delete them from th erunning_vars
    keys_to_remove.clear() #clear key list to prevent potential future issues with the list.
                
    # Loop through keys and values in variables_param
    for key, dict_value in variables_param_copy.items():
        if key not in running_vars: #check if key exists
            running_vars[key] = {}  # Create key if it doesn't exist in running_vars
        if "mean" in dict_value:  # Check if "mean" exists in the dictionary
            running_vars[key]["mean"] = dict_value["mean"]  # Copy the mean value

# FUNCTION TO RUN A FULL SET UP

def machine_set(
        current_op, # current_op values are read only, a copy is not necessary.
        machine_id, # machine ID identifies which machine the operator is setting up. This is read only, no copies required.
        part_type, # part type could be an input from a previous command as the machine set function will update which part is set.
        set_bool, #temp variable from start up sequence, highlights when the machine does not have running_var values
        machine_temp_vars, # provides access to multiple machine dictionaries which can be copied or linked as necessary.
        machine_values_copy # sub function modifies the values, so always link a copy and NOT the master values
        ):
    
    #machine attributes
    if "tooling_param" not in machine_temp_vars:
             machine_temp_vars["tooling_param"] ={
                 "tool_health": 0,
                 "tool_nom_dev": 0
             }
    tool_health = machine_temp_vars["tooling_param"]["tool_health"] #direct link so running tool health is changed
    if "set_part" not in machine_temp_vars["running_vars"]:
        machine_temp_vars["running_vars"]["set_part"] = "NONE"
    current_set_part = machine_temp_vars["running_vars"]["set_part"] # find the part that was last set up on this machine.
    set_up_time = machine_values_copy["set_up_time"]
    machine_start_t = machine_values_copy["start_up_time"]
    machine_state = machine_temp_vars["machine_state"]
    variables_param_copy = machine_temp_vars["variables_param"][part_type].copy()
    running_vars = machine_temp_vars["running_vars"][part_type]

    #operator attributes
    op_name = current_op["op_name"] #define variable for variable operator name
    judgetime = current_op["op_judge_time"] #define variable for variable operator judgement time
    op_set_multi = current_op["op_set_multi"] #define operator setting multiplier
    op_skill_high = current_op["op_skill_high"] #define operator skill level

    #unique variables
    settime = set_up_time*op_set_multi #define variable for operator setting time

    print(f"{op_name} is checking to see if {machine_id} has tooling and if it is set")
    time.sleep(judgetime)
    
    if set_bool == False:
        print(f"{op_name} has finished checking {machine_id} and established that it needs to be set completely") # GUI print to inform observer what the operator is doing.
        time.sleep(settime) # delay simulates operator setting time

        tool_change( #call the tool change function to update tool health, which will also reset means and std_devs showing the operator has realigned the tooling.
            machine_id=machine_id, # feed machine_id through, for GUI responses and running_vars dictionary definition, copies not necessary
            machine_values_copy=machine_values_copy, # feed machine_values copy through
            machine_temp_vars=machine_temp_vars, #allows function to change these values
            current_op=current_op, # feed current_op through, copies not necessary
            variables_param_copy=variables_param_copy, # feed variables through, read only, copies unnecessary
            )
        
    else:
        if current_set_part == part_type: # check to see if the part being set matches the previous part that was set. Running the same part type concurrently can save time setting up.  
            print(f"{op_name} has identified part {part_type} is already set on machine {machine_id}") # GUI print to inform observer what the operator is doing.
            print(f"{op_name} is inspecting the tooling") # GUI print to inform observer what the operator is doing.
            time.sleep(judgetime) # delay to simulate operator judging the tooling
            if op_skill_high > 60: #higher skilled operators will let the tool run longer to save money. This does save money in tooling, but also increases risk of 
                skill_issue = random.randint(15,25) # skilled operators will let machines run the tooling down further and have a narrow decision making window
            else: 
                skill_issue = random.randint(5,50) # less skilled operators are less likely to choose the best time to change tools
            if tool_health < skill_issue: # Check if the tool is healthy. 
                print(f"{op_name} has decided it would be best to change the tool inserts before starting the machine") # GUI print to inform observer what the operator is doing.
                time.sleep(settime) # delay to simulate the machine tools being changed
                
                tool_change( #call the tool change function to update tool health, which will also reset means and std_devs showing the operator has realigned the tooling.
                    machine_id=machine_id, # feed machine_id through, for GUI responses and running_vars dictionary definition, copies not necessary
                    machine_values_copy=machine_values_copy, # feed machine_values copy through
                    machine_temp_vars=machine_temp_vars, #allows function to change these values
                    current_op=current_op, # feed current_op through, copies not necessary
                    variables_param_copy=variables_param_copy, # feed variables through, read only, copies unnecessary
                    )
            else:
                print(f"{op_name} thinks the tools look ok. They have decided to realign targets and run the machine") #GUI print for observer awareness
                time.sleep(settime/4) #shorter set time to represent the easy changeover
                set_means(
                    variables_param_copy=variables_param_copy, #root copy passed through
                    running_vars=running_vars #direct link to make changes
                )
        else:
            print(f"{op_name} has checked if {machine_id} settings match {part_type}. They do not, so {machine_id} needs to be set.")
            print(f"{op_name} has started to set machine {machine_id}") # GUI print to inform observer what the operator is doing.
            time.sleep(settime) # delay simulates operator setting time

    machine_temp_vars["running_vars"]["set_part"] = part_type # update the temporary set part value held in the machine.

    #code to update Machine control SQL goes here
    print(f"{op_name} has started Machine {machine_id} running parts {part_type}. The machine is currently spinning up.") # GUI print to let observer know that setting has finished.
    machine_temp_vars["machine_state"] = True
    machine_state = machine_temp_vars["machine_state"]
    time.sleep(machine_start_t) # delay to simulate machine warming up, getting to speed or settling.
    time_now = time.time()
    
    return time_now, machine_state

#------------------------
# MACHINE ISSUE FUNCTIONS
#------------------------

# FUNCTION TO CHANGE RUNNING VARIABLES WHEN THE MACHINE TOOL IS BROKEN

def tool_broken( #This function will apply harsh discrepencies to the machine running variables that feed the outputs, it should only be run once when the tool breaks as it will maximise the nominal std_dev for the tool and looping will cause exponential increases
        machine_temp_vars # direct link to machine temp vars for the machine_id. values will be updated.
        ):
    running_vars = machine_temp_vars["running_vars"] # direct link to running_vars to update stored values
    nominal_dev = machine_temp_vars["tooling_param"]["tool_nom_dev"] # direct link to update nominal dev stored in tool temp var dictionary.
    for key, dict_value in running_vars.items(): #find instances of std_dev in affected operations
        #update std_dev to represent tool wear
        if "std_dev" in dict_value: #check for std_dev values in the part running vars
            nominal_dev[key]["std_dev"] = nominal_dev[key]["std_dev"]*3 # update the nominal dev held in the running tooling variables to be 3* bigger.
            dict_value["std_dev"] = nominal_dev[key]["std_dev"] # set part production running vars to the new std_dev
        # add random offset to the mean to simulate breakage profile.
        elif "mean" in dict_value: # check for mean values in the part production running vars
            current_mean = dict_value["mean"] #set the current held mean as a variable
            current_dev = nominal_dev[key]["std_dev"] #find the "std_dev" associated with this part of the operation
            random_float = random.uniform(-1*current_dev,current_dev) # create a random drift within the maximum distribution of the machine in its broken state
            new_mean = current_mean + random_float # create variable new_mean from current mean and the random float
            dict_value["mean"] = new_mean # set the held value for the mean to its new offset

# FUNCTION TO SIMULATE MAJOR MACHINE DAMAGE

def major_damage( # this function permanaently alters the default std_dev of the machine simulating machine wear after a crash, it can be run multiple times, until the machine is seen as no longer fit for use.
        machine_temp_vars, # direct link to machine temp vars for the machine_id. values will be updated.
        machine_attributes # direct link to change the machine forever
        ):
    init_params_copy = machine_temp_vars["init_params"].copy()
    variables_param = machine_temp_vars["variables_param"] # direct link to variables_param to update stored values
    machine_tools = machine_attributes["select_tool"][tool] #Accessing the dictionary in this way should make std_dev adjustments affect the Machine Attributes table. In this way machine wear will stay with the machinein the variables table.
    breakdown_hz = machine_attributes["breakdown_hz"]
    machine_tools_copy = machine_attributes["select_tool"][tool]

    for key, dict_value in machine_tools["devs"].items(): #find instances of std_dev in affected operations
        #update std_dev to represent tool wear
        if "std_dev" in dict_value: #check for std_dev values in the part running vars
            current_dev = dict_value["std_dev"] 
            new_dev = current_dev + (current_dev*0.02)
            dict_value["std_dev"] = new_dev # set part production running vars to the new std_dev
    
    tool_break_chance = machine_attributes["breakdown_table"]["t_b"]["chance"] 
    tool_break_chance = tool_break_chance + tool_break_chance * 0.1 # increase chances of tool breaking by 10%
    machine_attributes["breakdown_table"]["t_b"]["chance"] = tool_break_chance 
    machine_attributes["breakdown_table"]["t_b"]["repair_difficulty"] += 5 #increase repair difficulty by 5. If repair_difficulty exceeds 100 then it will no longer be repairable.
    
    loose_lever_chance = machine_attributes["breakdown_table"]["s_d"]["chance"] 
    loose_lever_chance = loose_lever_chance + loose_lever_chance*0.1 # increase chances of tool breaking by 10%
    machine_attributes["breakdown_table"]["s_d"]["chance"] = loose_lever_chance 
    machine_attributes["breakdown_table"]["s_d"]["repair_difficulty"] += 5 #increase repair difficulty by 5. If repair_difficulty exceeds 100 then it will no longer be repairable.

    breakdown_hz += 1 # incrementing by 1 will slowly make the machine unfit to run as errors keep happening. it represents the machine "falling apart"
    machine_attributes["breakdown_hz"] = breakdown_hz

    master_params( # This function should always create variables_param dictionaries for the machine in question. It represents a part change and a tool change as both means and std_devs are affected.
            init_params=init_params_copy,
            machine_tool=machine_tools_copy,
            machine_temp_vars=machine_temp_vars #feed through to preserve machine id
            )

# FUNCTION TO SIMULATE MACHINE ERRORS

def machine_error( # This function will take the defined error type from the main code, change the machine state as necessary to stop the machine running if the error stops the machine.
        #This function will also apply changes to variables from more major crashes and represent increased machine wear related to number of instances of issues occuring.
        error_type, #error type will be the key "l_e" for load error, "t_b" for tool breakage or "s_d" for unpredictable drift. More may be added as code develops.
        machine_temp_vars, # direct link to machine temp params to make changes include machine_id key
        machine_attributes, # direct link to machine attributes to make changes
        part_type, #used for defining the correct running vars table
        machine_id, # read only for GUI
        current_op # read only for GUI
        ):
    
    #operator variables
    op_name = current_op["op_name"] # current operator's name
    reacttime = current_op["op_react_time"] # time it takes operator to react to things
    
    #machine variables
    breakdown_table = machine_attributes["breakdown_table"][error_type]
    tool_health = machine_temp_vars["tooling_param"]["tool_health"]
    nominal_dev = machine_temp_vars["tooling_param"]["tool_nom_dev"]
    running_vars = machine_temp_vars["running_vars"][part_type]

    error_list= [
        f"{machine_id} makes a loud noise, but no one seems to have noticed yet...",
        f"Is {machine_id} supposed to be smoking like that?",
        f"You are sure that {machine_id} didn't make that cyclic clicking noise before",
        f"{machine_id} sounds more and more like a NIN concert every day..."
    ]

    big_error_list = [
        f"with a loud shriek and a bang {machine_id} grinds to a halt. That sounded expensive.",
        f"{machine_id} starts whining loudly, and flames start rising from the tool head! {op_name} runs to hit the E-Stop",
        f"{machine_id} starts spraying oil all over the factory! {op_name} rushes to hit the E-stop"
    ]
    
    if error_type == "l_e": # check to see if the error is a load error
        machine_temp_vars["machine_state"] = False # machine stops running
        machine_state = machine_temp_vars["machine_state"] #update variable to represent change
        time.sleep(reacttime) #simulate the time for the operator to notice the machine has stopped working.
        breakdown_table["chance"] += 1
        print(f"{op_name} has noticed that machine {machine_id} has stopped running") # GUI print so observers know when the operator notices the machine has stopped.
    elif error_type == "t_b": #check to see if their is a tool breakage error
        if tool_health == 0: # if there is a tool breakage error whilst the tool_health is at 0 the machine will crash, make a loud noise and will be stopped.
                #If the machine has been allowed to run with the tool broken, there is a chance that a major breakdown could occur. 
                #If a major breakdown does not occur then at least the machine will not work on the parts anymore
                # since we don't have input sizes yet we will add that later and just make the values mean +2 and std_dev +10 to roughly simulate initial processing
                major_damage(
                    machine_temp_vars=machine_temp_vars,
                    machine_attributes=machine_attributes
                )
                error_msg = random.choice(big_error_list)
                print(error_msg) #GUI to show that the machine has stopped.
                breakdown_table["chance"] += 1
                machine_attributes["breakdown_table"]["s_d"]["chance"] += 1
                machine_temp_vars["machine_state"] = False # machine stops running
                machine_state = machine_temp_vars["machine_state"] #update variable to represent change
        else: # The first time their is a tool breakage error the tool health is set to zero and the running variables are maxed
            machine_temp_vars["tooling_param"]["tool_health"] = 0
            tool_broken(
                machine_temp_vars=machine_temp_vars
            )
            error_msg = random.choice(error_list)
            print(error_msg)
    elif error_type == "s_d": # error type is s_d this is a standard drift, note repair is not called, it must be detected by a skilled operator or by an SPC  procedure (not yet coded)
        random_key = random.choice(list(running_vars.keys())) # select a random key from the list of operations, examples include: diameter1, length3, etc
        current_mean = running_vars[random_key]["mean"] # pull the current mean for the random operation
        current_dev = nominal_dev[random_key]["std_dev"] #find the "std_dev" associated with this part of the operation
        random_float = random.uniform(-2*current_dev,2*current_dev) # create a random drift within the 2/3rds of maximum distribution of the machine in its broken state
        new_mean = current_mean + random_float # create variable new_mean from current mean and the random float
        running_vars[random_key]["mean"] = new_mean #update the running vars to reflect the new mean
        breakdown_table["chance"] += 1
    
    return error_type

def repair(
        error_type, # feed error type to customise responses to different error types
        current_op, # feed the function the table related to the current operator. this is read only, so no copy is necessary.
        machine_id, # feed the function the relevant machine_id, this will only be used for prints and directory enquiries, so no copy is necessary
        tool,
        machine_attributes,# feed the function the breakdown table in the machine attributes dictionary specific to this machine.
        machine_temp_vars # used to access variables tables, direct link necessary.
        ):
    counter = 0 # counter to store operator attempts, different operators may try a different number of times, based on their experience.

    #machine variables direct links
    breakdown_table = machine_attributes["breakdown_table"][error_type]
    machine_state = machine_temp_vars["machine_state"] #direct link to make updates
    repair_difficulty = breakdown_table["repair_difficulty"] # defines the difficulty of this repair.

    #fixed variables (copied)
    init_tool_health = machine_attributes["select_tool"][tool]["init_tool_health"]
    rep_time = breakdown_table["fix_time"] # standard time to fix this issue
    init_tool_health_copy = init_tool_health
    rep_time_copy = rep_time

    #operator variables
    op_name = current_op["op_name"] # current operator's name
    op_judge_time = current_op["op_judge_time"]
    op_skill_low = current_op["op_skill_low"] # lower threshold of operator skill, defines possibility of operator fumbling tools and taking twice as long.
    op_skill_high = current_op["op_skill_high"] # upper threshold of operator skill, defines whether the operator is capable of making this repair
    op_set_multi = current_op["op_set_multi"] # multiplier for operators speed at fixing and setting
    tenacity = current_op["tenacity"] # tenacity defines how many times the operator will attempt a repair before the give up and ask for help.

    op_rep_time = rep_time_copy*op_set_multi # set the variable defining current operators time to make this repair
    counter = 0 # tracks number of repair attempts the operator has made

    # Function begins
    print(f"{op_name} has started attempting to fix {machine_id}")
    machine_temp_vars["machine_state"] = False # machine stops running
    machine_state = machine_temp_vars["machine_state"] #update variable to represent change

    fail_list = [
        f"{op_name} can't seem to figure out the problem",
        f"{op_name} scratches their head, they were sure that should have worked",
        f"{op_name} is sure they nearly had it that time",
        f"{op_name} is rooting around their toolbox for the 'thingamajig'"
    ]
    
    epic_fail_list = [
        f"You hear a crash as {op_name} fumbles something deep in the machine, repairs will now take longer.",
        f"{op_name} shouts at the machine. It's not going well",
        f"CLANG! {op_name}'s spanner flies across the floor. They seem pretty angry.",
        f"A kindly colleague brings {op_name} a cup of tea. They look like they need it."
    ]

    while machine_state == False: #loop until the machine is operational
        if counter < tenacity: # check the count of successful attempts to see if the operator is tenacious enough to keep attempting repairs   
            if counter > 7:
                print(f"Tenacious {op_name} decides to have another go at the repairs")
            elif counter > 0:
                print(f"{op_name} decides to have another go at the repairs")
        else: #exception for when the operator has exhausted their tenacity and assumes there is a skill issue
                    fail_message = random.choice(fail_list)
                    print(fail_message)
                    time.sleep(op_judge_time)
                    print(f"{op_name} has tried everything they can, they are now looking for a craftsperson or engineer for assistance") # GUI print out to show operator has given up on repairs
                    break # exit loop
        repair_attempt = random.randint(1,100) # generate a random number to simulate the operator attempting to fix the error
        time.sleep(op_rep_time) # delay represents time operator takes to attempt the repair
        if repair_attempt < op_skill_low: # check if the repair attempt results in a fumble
            epic_fail_message = random.choice(epic_fail_list)
            print(epic_fail_message) # GUI print so observers know the operator has fumbled
            op_rep_time *= 2 # repairs now take twice as long since the operator has made matters worse.
            tenacity *= 1.1 # tenacity increase represents operator embaressment for the fumble and their belief that they can fix their own mistakes
            counter += 1
        elif repair_attempt < repair_difficulty: # check to see if the repair attempt is not good enough, but not a fumble
            fail_message = random.choice(fail_list)
            print(fail_message)
            counter += 1 # increment the number of attempts
        elif repair_attempt > repair_difficulty: # check to see if the repair attempt copuld be successful
            if op_skill_high < repair_difficulty: # check to see if this operator is skilled enough to successfully make the repair.
                    counter += 1 # increment the number of attempts
                    fail_message = random.choice(fail_list)
                    print(fail_message) #GUI print to show the operator attempted to fix the machine
            else: #condition: operator is skilled enough to make the repair
                if error_type == "l_e":
                    print(f"{op_name} has successfully repaired the machine! It was just a misloaded part.") #GUI print to make observer aware that 
                if error_type == "t_b":
                    print(f"{op_name} has successfully repaired the machine It was a fairly difficult repair, but they did it!")
                    machine_temp_vars["tooling_param"]["tool_health"] = init_tool_health_copy
                machine_temp_vars["machine_state"] = True # machine is operational again
                machine_state = machine_temp_vars["machine_state"] #update variable to represent change
                if breakdown_table["chance"] > 20: # check to see that the error has a significant chance of occuring
                    if repair_attempt > repair_difficulty + 10: # repair difficulty +10 specifies that this attempt is a particularly good attempt.
                        breakdown_table["chance"] /= 2 # reduces the chance of this error occuring again significantly
                        print(f"{op_name} did a particularly good job on this repair, it is less likely to happen in future")
                        if op_skill_high < 50: #check operator skill level
                            current_op["op_skill_high"] += 0.5 #increase skill
                        elif op_skill_high < 90:
                            current_op["op_skill_high"] += 0.1 # skill increase diminishing return
                        elif op_skill_high < 100:
                            current_op["op_skill_high"] += 0.01 # skill increase diminishing return

#--------------------------
# MACHINE RUNNING FUNCTIONS
#--------------------------

# FUNCTION TO COMPLETE CHECKS, SET AND RUN MACHINE

#Check init_params exist

def initial_checks(
        current_op,
        machine_id,
        part_type,
        machine_temp_vars,
        machine_attributes,
        part_attributes
        ):
    
    machine_tools = machine_attributes["select_tool"][tool] #direct link for updates
    drawing_specs = part_attributes["target_dims"].copy() #copy to preserve drawing values

    #check if machine state has been declared previously
    if "machine_state" not in machine_temp_vars:
        machine_temp_vars["machine_state"] = False

    #check init_params exist
    if "init_params" not in machine_temp_vars:
        machine_temp_vars["init_params"] = {}
    if part_type not in machine_temp_vars["init_params"]:
        machine_temp_vars["init_params"][part_type] ={}
        # INIT_PARAMS_CREATION
        create_inits(
        drawing_specs=drawing_specs, #drawing specs drawn from part_attribute table.
        machine_temp_vars=machine_temp_vars
        )

    #check variables_param exist
    
    if "variables_param" not in machine_temp_vars:
        machine_temp_vars["variables_param"] = {}
    if part_type not in machine_temp_vars["variables_param"]:
        machine_temp_vars["variables_param"][part_type] = {}
        # CREATE VARIABLES_PARAM FROM MASTERS

        init_params_copy = {} #create/clear a dictionary to copy the init_params into
        machine_tools_copy = {}

        init_params_copy = machine_temp_vars["init_params"][part_type].copy() #copying the dictionary in this way should allow us to link to the variable params and make updates without affecting the init_params
        machine_tools_copy = machine_tools.copy()

        master_params( # This function should always create variables_param dictionaries for the machine in question. It represents a part change and a tool change as both means and std_devs are affected.
            init_params=init_params_copy,
            machine_tool=machine_tools_copy,
            machine_temp_vars=machine_temp_vars #feed through to preserve machine id
            )
    
    if "running_vars" not in machine_temp_vars:
        machine_temp_vars["running_vars"]={}
        set_bool = False
    else:
        set_bool = True
    if part_type not in machine_temp_vars["running_vars"]:
        machine_temp_vars["running_vars"][part_type] = {}
        set_bool = False
    else:
        set_bool = True

    if "part_id" not in machine_temp_vars["running_vars"]: # this will always be true until SQL is set up to integrate with this code.
        machine_temp_vars["running_vars"]["part_id"] = {}
        part_id = machine_temp_vars["running_vars"]["part_id"][part_type] = 0
    elif part_type not in machine_temp_vars["running_vars"]["part_id"] or not machine_temp_vars["running_vars"]["part_id"][part_type]:
        part_id = machine_temp_vars["running_vars"]["part_id"][part_type] = 0
    else:
        part_id = machine_temp_vars["running_vars"]["part_id"][part_type]

    low_score = part_id

    with operator_lock:

        # RESET RUNNING VARIABLES TO DRAWING AND MACHINE CAPABILITY

        machine_values_copy = {} #create/empty the machine_values temporary dictionary

        machine_values_copy = machine_attributes.copy() #copy machine_values from the Master attributes dictionary             

        start_time, machine_state = machine_set( # Function to simulate an operator performing a tool change on the machine. means and std-devs are affected.
            current_op=current_op, #link to operator table
            machine_id=machine_id, # machine_id is provided for GUI read only. No copies needed
            part_type=part_type, #link to part type
            set_bool=set_bool,
            machine_temp_vars=machine_temp_vars, # provides access to multiple machine dictionaries which can be copied or linked as necessary.
            machine_values_copy=machine_values_copy
            )
        
    return start_time, machine_state, low_score
        
# FUNCTION TO REPRESENT MACHINE REGULAR MAINTENANCE

def mc_maint(
        maint_timer,
        current_op,
        machine_id,
        machine_values_copy,
        machine_temp_vars
        ):
        
        #Operator variables
        op_name = current_op["op_name"]
        op_react_time = current_op["op_react_time"]
        op_judge_time = current_op["op_judge_time"]
        op_skill_high = current_op["op_skill_high"]

        #machine attributes
        tool_health = machine_temp_vars["tooling_param"]["tool_health"]
        variables_param_copy = machine_temp_vars["variables_param"][part_type].copy()

        if time.time() - maint_timer > op_react_time*2: # We will add prio stacking here once we have integrated threading
            print(f"{op_name} decides to check tooling and maintenance on {machine_id}") # GUI print to let observer know that maintenance is occuring
            with machine_vars_lock:
                time.sleep(op_judge_time) # delay to simulate operator stopping the machine and checking the tooling
                if op_skill_high > 60: #check operator skill level
                    skill_issue = random.randint(15,25) # skilled operators will let machines run the tooling down further and have a narrow decision making window
                else: 
                    skill_issue = random.randint(5,50) # less skilled operators are less likely to choose the best time to change tools
                if tool_health < skill_issue: # check to see if the current tool health prompts the operator to feel a change is necessary
                    if tool_health == 0: #check to see if the tooling is in really bad condition - if this happens often it will indicate that maintenance needs to be more regular
                        print(f"It looks like {op_name} has discovered some major tooling damage on machine {machine_id}. Good job they caught it before disaster struck!") # GUI print to let observers know the operator is a hero
                    tool_change( #call the tool change function to update tool health, which will also reset means and std_devs showing the operator has realigned the tooling.
                        machine_id=machine_id, # feed machine_id through, for GUI responses and running_vars dictionary definition, copies not necessary
                        machine_values_copy=machine_values_copy, # feed machine_values copy through
                        machine_temp_vars=machine_temp_vars, #allows function to change these values
                        current_op=current_op, # feed current_op through, copies not necessary
                        variables_param_copy=variables_param_copy, # feed variables through, read only, copies unnecessary
                        )
                else:
                    print(f"{op_name} has decided the tooling on {machine_id} looks fine.")
                maint_timer = time.time()

        return maint_timer

def error_check(
        machine_attributes, # direct link for errors that cause permanent damage, can be copied for other uses
        machine_temp_vars, # direct link to update variables tables as necessary
        part_type, #used for dictionary directories only
        machine_id, # used for dictionary directories only
        current_op # used for dictionary directories only
        ):
    
    # machine attributes
    breakdown_hz = machine_attributes["breakdown_hz"]
    breakdown_table = machine_attributes["breakdown_table"]
    

    error_type = ""

    #program begins
    error = random.randint(1,100) #generate a random number to see if the machine has an error
    if error < breakdown_hz: #check our random number against the machine breakdown frequency
        errorlist = [] #generate a list to populate with possible errors
        for key, value in breakdown_table.items():
            errorlist.extend([key] * int(value["chance"])) #populate error list with error types, weighted by their chance of occuring
        error_type = random.choice(errorlist) #choose a weighted random error type from the created list

        error_type = machine_error(
            error_type=error_type,
            machine_temp_vars=machine_temp_vars,
            machine_attributes=machine_attributes,
            part_type=part_type,
            machine_id=machine_id,
            current_op=current_op
        )
    
    return error_type
        

# FUNCTION TO SIMULATE TOOL WEAR

def tool_wear( # Function to simulate tool wear, mean is moved a small amount and at low tool health std_dev is altered.
        temp_running_vars, # values are changed here. This should be linked to the running_vars table for the machine_id in the machine_temp_vars
        tool_vars, # direct link to the machine_temp_vars tooling table for this machine_id
        temp_init_tool_health # This values is referenced for comparison only, a copy is not necessary.
        ):
    if isinstance(tool_vars["tool_health"], int) and tool_vars["tool_health"] >= 0: #error handling
        if isinstance(temp_init_tool_health, int) and temp_init_tool_health >0: #error handling
            health_percentage = tool_vars["tool_health"] / temp_init_tool_health # Checks the variable tool health against the initial tool health to provide a percenbtage score
            if health_percentage < 0.1: # Check Tool health for significant wear
                for key, dict_value in temp_running_vars.items(): #find instances of std_dev in affected operations
                    #update std_dev to represent tool wear
                    if "std_dev" in dict_value: # find running var std_dev values
                        current_dev = dict_value["std_dev"] # define the current std_dev as a variable
                        new_dev = current_dev + (current_dev * 0.05) # create the new std_dev as a percentage change of the old std_dev. over 10 iterations this percentage change would change a total distribution of 3 to 4.88 which seems fair representing the degredationm of the tool
                        dict_value["std_dev"] = new_dev # apply the new std_dev to the running_vars table

            if health_percentage < 0.95: #Check Tool health ignoring minimal wear
                for key, dict_value in temp_running_vars.items(): #find instances of mean in affected operations
                    #update mean to represent tool wear
                    if "mean" in dict_value: # find running_var mean values
                        current_mean = dict_value["mean"] # set the current mean as a variable
                        current_drift = tool_vars["tool_nom_dev"]*6/100 #create the current drift from the expected tool std_dev multiplied by 6  to represent the total distribution and divided by 100 to make it a 1% of the total distribution drift, representing tool material that is removed without chipping whilst working the parts.
                        new_mean = current_mean + current_drift # henerate the new mean by adding the drift to the old mean
                        dict_value["mean"] = new_mean # update the running_vars mean
            if health_percentage > 0: # check that tools have a health above 0 as tools can not fall below 0 health without causing errors.
                tool_vars["tool_health"] -= 1 # negatively increment tool health by one.
        else:
            print("Error! initial tool health is 0, negative or not a number") #error handling
            sys.exit() # terminate the programme to be updated when testing is complete for more robust error handling.
    else:
        print("Error! variable tool health is 0, negative or not a number") #error handling
        print(tool_vars["tool_health"])
        sys.exit() # terminate the programme to be updated when testing is complete for more robust error handling.

# FUNCTION FOR THE MACHINE TO CREATE AN OUTPUT

def part_work(
        machine_values_copy,
        machine_temp_vars,
        part_type,
        machine_id
        ):
    
    #Machine variables
    running_vars_copy = {}

    running_vars_copy = machine_temp_vars["running_vars"][part_type].copy()
    cycle_t = machine_values_copy["cycle_t"] # cycle time does not need to be altered by this function

    machine_temp_vars["running_vars"]["part_id"][part_type] += 1
    part_id = machine_temp_vars["running_vars"]["part_id"][part_type]
    
    part_load = 1 #simulates a part being loaded to the machine
    load_time = time.strftime("%d-%b-%Y %H:%M:%S",time.localtime(time.time())) #formatting load time for database analysis
    time.sleep(cycle_t) #time.sleep represents the cycle time of the machine
    output_values = make_stuff( #seperating machine outputs into a different function
            running_vars_copy=running_vars_copy
            ) 
    sql_string, print_string = print_stuff(
            output_values=output_values
            )
    print(f"INSERT INTO {machine_id} (PartID, Timestamp, Partload, {sql_string}) \n"\
    f"\tVALUES (P{part_id:05}, {load_time}, {part_load}, {print_string})")

    #CODE TO APPLY TOOL WEAR
    temp_running_vars = {}
    temp_init_tool_health = {}

    temp_running_vars = machine_temp_vars["running_vars"]
    tool_vars = machine_temp_vars["tooling_param"]
    temp_init_tool_health = machine_values_copy["select_tool"][tool]["init_tool_health"]

    tool_wear(
        temp_running_vars=temp_running_vars,
        tool_vars=tool_vars,
        temp_init_tool_health=temp_init_tool_health
        )
        
    

# FUNCTION TO CREATE PARTS IN A STANDARD DISTRIBUTION

def make_stuff(
        running_vars_copy # feed the function the running_Vars specific to the part_type. These should be copies to prevent unwanted running_Var deviation.
        ):
    
    output_values = {} # clear/ empty output values dictionary for use.

    keys_list = [key for key in running_vars_copy]
    for key in keys_list: # cycle through each operation defined by the part type
        mean = running_vars_copy[key]["mean"]
        std_dev = running_vars_copy[key]["std_dev"]
        value = np.random.normal(mean, std_dev) # for each operation, generate a value from the normal distribution available.
        output_values[key] = value # enter each operational value into the output_values dictionary.

    return output_values # return output values to the previous function.

# FUNCTION TO CREATE STRINGS FOR SQL QUERIES

def print_stuff(output_values): #generated by the make_stuff function
    print_string = [] # clear/empty the print string list
    sql_string = [] # clear/empty the SQL string list
    for key, value in output_values.items(): #cycle each value in the output values dictionary by key and value.
            sql_string.append(key)
            print_string.append("{:.4f}".format(value)) # add the values to the print string.
    return ", ".join(sql_string), ", ".join(print_string) #return the string lists as a single string separated by commas.

# RUNNING SEQUENCE - NO THREADS

def machine_root(
    machine_id,
    current_op,
    part_type,
    tool,
    machine_temp_vars,
    machine_attributes,
    part_attributes
    ):

    #machine variables
    run_time = machine_attributes["run_time"] #copied to prevent alterations to run time.
    run_time_copy = run_time

    #Operator variables
    op_name = current_op["op_name"]

    with machine_vars_lock: #lock machines thread to prevent shared variables getting mixed up
        with attributes_lock: #lock attributes thread to prevent shared variables getting mixed up

            start_time, machine_state, low_score = initial_checks(
                current_op=current_op,
                machine_id=machine_id,
                part_type=part_type,
                machine_temp_vars=machine_temp_vars,
                machine_attributes=machine_attributes,
                part_attributes=part_attributes
                )
    
    maint_timer = time.time() #timer to space out operator maintenance.

    machine_values_copy = {}

    machine_values_copy = machine_attributes.copy() #mc_maint function should not alter the machine attributes
    
    while machine_state == True and time.time() - start_time < run_time_copy: # using run time to make this program run for a number of seconds for now, but later we can tie this to parts on order or operator shift/break times
        
        with operator_lock:
            
            maint_timer = mc_maint(
                maint_timer=maint_timer,
                current_op=current_op,
                machine_id=machine_id,
                machine_values_copy=machine_values_copy,
                machine_temp_vars=machine_temp_vars
            )

        machine_state = machine_temp_vars["machine_state"]
        with machine_vars_lock:
            error_type = error_check( # the variety of errors could update any of these inputs
                    machine_attributes=machine_attributes,
                    machine_temp_vars=machine_temp_vars,
                    part_type=part_type,
                    machine_id=machine_id,
                    current_op=current_op
                    )
            
        machine_state = machine_temp_vars["machine_state"]

        with machine_vars_lock:
            
            if machine_state == False:
                #when threading is set up we will add task priority here
                with operator_lock:
                    machine_state = repair(
                        error_type=error_type,
                        current_op=current_op,
                        machine_id=machine_id,
                        tool=tool,
                        machine_attributes=machine_attributes,
                        machine_temp_vars=machine_temp_vars
                    )

                    maint_timer = time.time()

                    machine_state = machine_temp_vars["machine_state"]
                
                    if machine_state == False:
                        print(f"{op_name} has given up on {machine_id}. It shall remain idle until it receives repairs from someone more skilled")
                        part_id = machine_temp_vars["running_vars"]["part_id"][part_type]
                        score_calc = part_id
                        high_score = score_calc - low_score
                        print(f"Thanks for playing, you and {op_name} made {high_score} parts!")
                        break # exit loop


            elif machine_state == True:
                #Implement an SPC step here for skilled operators or a paper/automatic  SPC procedure, this will need to feed from SQL once implemented to read the last x parts output from the machine.
                # spc_check()

                part_work(
                    machine_values_copy,
                    machine_temp_vars,
                    part_type,
                    machine_id
                    )
        machine_state = machine_temp_vars["machine_state"]

    if machine_state == True:
        print(f"{op_name} has gotten to the end of their shift and has stopped machine {machine_id} and started cleaning down the area")
        part_id = machine_temp_vars["running_vars"]["part_id"][part_type]
        score_calc = part_id
        high_score = score_calc - low_score
        print(f"Thanks for playing, you and {op_name} made {high_score} parts!")

part_type = "DRW001" # Placeholder for functionality, to be derived by Factory Manager/Operator part assignment.
machine_id = "MC001"
tool = "standard"
op_id = "OP01"

current_op = operators[op_id]

machine_root(
    machine_id=machine_id,
    current_op=current_op,
    part_type=part_type,
    tool=tool,
    machine_temp_vars=machine_temp_vars[machine_id],
    machine_attributes=machine_attributes[machine_id],
    part_attributes=part_attributes[part_type]
    )
#    print(f"{op_name} has gotten to the end of their shift and has stopped machine {machine_id} and started cleaning down the area")
  
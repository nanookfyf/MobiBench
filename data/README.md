---
license: apache-2.0
language:
- zh
---
# Chinese-Mobile-Use

For details, please refer to our [repo](https://github.com/ipxx-SAI/MobiAgent) 

You need to decompress `rawdat.tar.gz` and place it in the `rawdata` directory, as shown in the example given.

## Data Format

`data_path` stores the main dataset with the following directory structure:

```
data/
|-- some-subpath1
|   |-- 1.jpg
|   |-- 2.jpg
|   |-- ...
|   |-- actions.json
|   `-- react.json
`-- some-subpath2
    |-- 1.jpg
    |-- 2.jpg
    |-- ...
    |-- actions.json
    `-- react.json
```

Each deepest sub-directory stores a complete trajectory consisting of screenshots and actions, in which `actions.json` stores the low-level action trajectory with the following format:

```
{
  "app_name": string,
  "task_description": ["The description of the task list."],
  "action_count": "The count of the actions.",
  "actions": [
    {
      "type": "The type of the action",
      "parameters": "etc."
    },
    {
      "type": "click",
      "position_x": "x-coordinate of click",
      "position_y": "y-coordinate of click action",
      "bounds": "the bound of the clicked element"
    },
    {
      "type": "swipe",
      "press_position_x": "x-coordinate of press",
      "press_position_y": "y-coordinate of press",
      "release_position_x": "x-coordinate of release",
      "release_position_y": "y-coordinate of release",
      "direction": "The direction of the user's swipe gesture. UP: swipe finger upward to scroll content up and reveal content below. DOWN: swipe finger downward to scroll content down and reveal content above. LEFT: swipe finger leftward to scroll content left. RIGHT: swipe finger rightward to scroll content right."
    },
    {
      "type": "input",
      "text": "The text to input"
    },
    { "type": "done" },
    { "type": "wait" }
  ]
}
```

and `react.json` stores the VLM-annotated high-level reasoning trajectory with the following format:

```json
[
  {
    "reasoning": "Reasoning for choosing this operation type",
    "function": {
      "name": "click",
      "parameters": {
        "target_element": "High-level semantic description of target"
      }
    }
  },
  {
    "reasoning": "Reasoning for swipe operation",
    "function": {
      "name": "swipe",
      "parameters": {
        "direction": "UP, DOWN, LEFT, RIGHT"
      }
    }
  },
  {
    "reasoning": "Reasoning for text input",
    "function": {
      "name": "input",
      "parameters": {
        "text": "Text to input"
      }
    }
  },
  {
    "reasoning": "Basis for task completion",
    "function": {
      "name": "done",
      "parameters": {}
    }
  },
  {
    "reasoning": "Reason for waiting",
    "function": {
      "name": "wait",
      "parameters": {}
    }
  }
]
```

`ss_data_path` stores manually collected single-step action data and can be empty. Example directory structure is as follows:

```
ss_data/
|-- decider
|   `-- some-subpath
|       |-- 1.jpg
|       |-- 2.jpg
|       |-- ...
|       |-- react.json
|       `-- tasks.json
`-- grounder
  `-- some-subpath
    |-- 1.jpg
    |-- 2.jpg
    |-- ...
    `-- react.json
```

`ss_data_path` must only contain `decider` and `grounder` as top-level directories, for training the respective models. `some-subpath` can be any depth or name. The deepest subdirectory contains `n` screenshots and action list `react.json` of length `n`, with actions and screenshots matched by index, and all pairs are single-step and independent. Subdirectories do not contain `actions.json`.

* Subdirectories under `ss_data/decider` also contain a `tasks.json` list of arbitrary length. When constructing the training dataset, a random task is sampled from the list for each screenshot-action pair to fill the task description part of the training prompt.
* Subdirectories under `ss_data/grounder` should have a `react.json` in which each item must be `click` action and include an additional `bbox` field representing the bounding box (absolute coordinates) of the target element, for example:

```json
[
    {
        "reasoning": "...",
        "function": {
            "name": "click",
            "parameters": {
                "target_element": "..."
            }
        },
        "bbox": [100, 200, 300, 400]
    }
]
```

`unexpected_img_path` directory stores screenshots of ads or pop-ups that require the agent to terminate task execution when encountered, and can be empty.
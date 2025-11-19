import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from collections import defaultdict

def plot_gantt_chart(timeline_events, title="Gantt Chart"):
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 1. Initialize data structures
    task_start_end = defaultdict(lambda: {'start': None, 'end': None})
    # Variable to track the latest completion time across all tasks
    absolute_latest_end_time = 0.0

    # 2. Iterate through the event list to find the start/end times and the latest completion time
    for event in timeline_events:
        task_name = event['task']
        timestamp = event['timestamp']
        event_type = event['event']
        
        # Standardizing event types (mapping specific ones like START_PIPELINE to START)
        if 'START' in event_type:
            # Only record the very first start time for a task
            if task_start_end[task_name]['start'] is None:
                task_start_end[task_name]['start'] = timestamp
                
        elif 'END' in event_type:
            # Record the latest end time for a task (if it has multiple ends)
            if task_start_end[task_name]['end'] is None:
                task_start_end[task_name]['end'] = timestamp
            else:
                task_start_end[task_name]['end'] = max(task_start_end[task_name]['end'], timestamp)
                
                # Track the overall latest completion time (important for main_pipeline alignment)
            if task_start_end[task_name]['end'] > absolute_latest_end_time:
                absolute_latest_end_time = task_start_end[task_name]['end']

    # --- NEW LOGIC: Ensure main_pipeline ends with the absolute last operation ---
    main_pipeline_name = 'main_pipeline' # Assuming this is the top-level task name
    
    if absolute_latest_end_time > 0 and main_pipeline_name in task_start_end:
        # Check if the pipeline's currently recorded end is earlier than the actual latest end time
        pipeline_end_time = task_start_end[main_pipeline_name].get('end')
        
        if pipeline_end_time is None or absolute_latest_end_time > pipeline_end_time:
            # Override the main_pipeline's end time with the absolute latest completion time
            task_start_end[main_pipeline_name]['end'] = absolute_latest_end_time
    # --------------------------------------------------------------------------

    # 3. Filter out tasks and finalize the lists
    tasks = []
    start_times = []
    end_times = []

    for task_name, times in task_start_end.items():
        if times['start'] is not None and times['end'] is not None:
            tasks.append(task_name)
            start_times.append(times['start'])
            end_times.append(times['end'])
    
    # Calculate duration (width of the bar)
    durations = [(end - start) for start, end in zip(start_times, end_times)]
    
    # Determine the task positions on the Y-axis
    y_positions = list(range(len(tasks)))
    
    # 2. Plot the horizontal bars
    ax.barh(y_positions, 
            durations, 
            left=start_times, 
            height=0.6, 
            align='center',
            color='skyblue') 

    # 3. Configure the plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(tasks)
    ax.invert_yaxis()  # To display tasks from top to bottom

    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    
    # Optional: Add gridlines for better readability
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
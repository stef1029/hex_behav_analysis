"""
This script will take the DLC data in each trial and extract features from it.
These features can then be examined and plotted, or passed to a clustering algorithm.
From chatGPT:
1. Path Efficiency or Directness
Straightness Index or Path Efficiency: This can be calculated as the ratio of the straight-line distance between the start and end points of the movement to the total path length traveled. A value close to 1 indicates a very direct path, while values farther from 1 indicate more circuitous routes.
Tortuosity: This measures the curvature or winding of the path. It can be computed as the total path length divided by the straight-line distance.
2. Speed Variations
Average Speed: As mentioned, the average speed throughout the trial. This is the total distance traveled divided by the total time taken.
Maximum and Minimum Speed: Peaks and troughs in the speed profile can indicate moments of acceleration or deceleration.
Speed Variability: Standard deviation or variance of speed over time, indicating the consistency of the mouse's speed.
3. Acceleration
Average Acceleration: Change in speed over time, averaged over the course of a trial.
Peak Acceleration and Deceleration: Maximum positive and negative changes in speed, showing the fastest gains and losses in velocity.
4. Angular Metrics
Angular Velocity: The rate of change of the angle of the movement path, which can indicate how quickly the mouse is turning.
Angular Change: Total sum of angular changes throughout the path, which can reflect the overall complexity of the movement.
5. Distance Metrics
Total Path Length: The cumulative distance the mouse travels during a trial.
Euclidean Distance (Straight-Line Distance): Distance between the start and end points of the path.
6. Stop-and-Go Behavior
Number of Stops: How many times the mouse stops during its movement, which can be inferred from when the speed drops to near zero.
Duration of Stops: Total time spent stopping, which can indicate hesitancy or exploration.
7. Area Covered
Enclosed Area: Area of the convex hull or the minimum bounding geometry that contains the entire movement path, indicating the space utilized during the movement.
"""
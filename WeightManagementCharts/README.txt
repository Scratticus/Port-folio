This folder contains some personal projects for weight management. The mass01 notebook takes weights recorded by a fitness app, creates a linear regression
from the daily records and plots the weights against the mean trend.

The notebook calculates the standard deviation of the weights and creates a Statistical Process Control (SPC) style Chart to monitor progress over the next time interval.
It also checks that weight loss efforts approach the target weight and returns the expected date that the target max weight will be achieved. Once the target weight is 
achieved the charts can become a stable SPC chart based on the target average weight std deviations.

This initial code, tracks all data to form an overview of progress. Future code will measure the months progress and create a modified SPC chart to maintain weight control
towards the target weight.

The SPC chart is in A4 format and ready for printing.
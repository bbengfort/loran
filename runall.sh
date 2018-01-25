#!/bin/bash

Y="recovery_time_mins"
YPOS="recovery_time_mins_pos"
YNEG="recovery_time_mins_neg"
EXCLUDE="building_id,date"

python3 loran.py -n 0 -e $EXCLUDE -o fixtures/esb-model-ranking.json -t $Y fixtures/esb-dataset.csv

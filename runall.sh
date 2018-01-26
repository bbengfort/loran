#!/bin/bash

Y="recovery_time_mins"
YPOS="recovery_time_mins_pos"
YNEG="recovery_time_mins_neg"
EXCLUDE="building_id,date"

python3 loran.py -n 2 -e $EXCLUDE -o fixtures/esb_model_ranking.json -t $YPOS fixtures/esb-dataset-pos.csv
python3 loran.py -n 2 -e $EXCLUDE -o fixtures/esb_model_ranking.json -t $YNEG fixtures/esb-dataset-neg.csv
python3 loran.py -n 2 -e $EXCLUDE -o fixtures/esb_model_ranking.json -t $Y fixtures/esb-dataset.csv

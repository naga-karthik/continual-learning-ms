#!/bin/bash

centers='amu bwh karo milan montpellier nih rennes ucsf mix'
for center in $centers; do
    python create_json_data.py -se 2 -ds $center
done 

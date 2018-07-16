import csv
import random
import numpy as np

results = []
with open("../data/jl_coffee_quality_roastingParams.csv") as csvfile:
    reader = csv.reader(csvfile) # change contents to floats
    for row in reader: # each row is a list
        results.append(row)
def load_data():
    raw_results = []
    with open("../data/jl_coffee_quality_roastingParams.csv") as csvfile:
        reader = csv.reader(csvfile) # change contents to floats
        for row in reader: # each row is a list
            raw_results.append(row)
    
    bean_types = []
    
    for row in raw_results[1:]:
        # Extract unique country names
        if not row[1] in bean_types:
            bean_types.append(row[1])
    
    results_in = []
    results_out = []
    
    for row in raw_results[1:]:
        input = []
        output = []
        
        for bean_type in bean_types:
            input.append(0.0)
        input[bean_types.index(row[1])] = 1.0
        input.append((float(row[2])-72.0)/400.0) # Initial temp
        input.append((float(row[3])-72.0)/400.0) # Min temp
        input.append(float(row[4])/750.0) # Min seconds
        input.append((float(row[5])-72.0)/400.0) # Max temp
        input.append(float(row[6])/750.0) # Max seconds
        input.append((float(row[7])-72.0)/400.0) # Final temp
        input.append(float(row[8])/750.0) # Seconds
        output.append((float(row[9])-77.0)/(89.0-77.0)) # Score
#        output.append(float(row[9])/100.0) # Score
#        output.append(float(row[9])) # Score
        input.append((float(row[10]))/13.0) # Roast time
        input.append((float(row[11]))/13.0) # Min minutes
        input.append((float(row[12]))/13.0) # Max minutes
        results_in.append(input)
        results_out.append(output)
        
    coffee_data = [
                    (np.asarray([float(s) for s in in_row]).reshape(len(in_row),1),
                    np.asarray([float(t) for t in out_row]).reshape(len(out_row),1)) 
                    for in_row,out_row in zip(results_in,results_out)
                ]
    
    random.shuffle(coffee_data)
    training_data = coffee_data[:60]
    validation_data = coffee_data[60:60]
    test_data = coffee_data[60:]
    
    return (training_data,validation_data,test_data)

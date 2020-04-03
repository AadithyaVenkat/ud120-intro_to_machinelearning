#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

#for key,value in enron_data.items():
#		 print(key, sum(1 for v in value if v))

count=0
for key in enron_data:
		if(enron_data[key]["poi"]==True):
				count+=1

print(count)


#print(enron_data["PRENTICE JAMES"]["total_stock_value"])

#print(enron_data["COLWELL WESLEY"]["from_this_person_to_poi"])

#print(enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])

"""
for key in enron_data:
		if(key == "SKILLING JEFFREY K" or key=="FASTOW ANDREW S" or key=="LAY KENNETH L"):
			print(key)
			print(enron_data[key]["total_payments"])
"""
"""
count=0
for key in enron_data:
		if(enron_data[key]["salary"] != "NaN"):
				count=count+1

print(count)
"""
"""
payment_count=0
for key in enron_data:
		if(enron_data[key]["total_payments"] == "NaN" and enron_data[key]["poi"]=="true"):
				payment_count+=1

print(payment_count)
"""

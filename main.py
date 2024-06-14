import tkinter as tk
from tkinter import messagebox
import XGBoostmodel
import pandas as pd

def predict_user_data():
    # DataFrame with user input
    data = {}
    for field, entry in entries.items():
        data[field] = entry.get()
    user = pd.DataFrame({
        'Drug' :[float(drug_var.get())],
        'Age':[float(data['Age'])],
        'Sex':[float(gender_var.get())],
        'Ascites':[float(ascites_var.get())],
        'Hepatomegaly' :[float(hepatomegaly_var.get())],
        'Spiders':[float(spiders_var.get())],
        'Edema':[float(edema_var.get())],
        'Bilirubin':[float(data['Bilirubin'])],
        'Cholesterol':[float(data['Cholesterol'])],
        'Albumin':[float(data['Albumin'])],
        'Copper':[float(data['Copper'])],
        'Alk_Phos':[float(data['Alk_Phos'])],
        'SGOT':[float(data['SGOT'])],
        'Tryglicerides':[float(data['Tryglicerides'])],
        'Platelets':[float(data['Platelets'])],
        'Prothrombin':[float(data['Prothrombin'])],
    })
    # Predict using the trained XGBoost model
    prediction = XGBoostmodel.predict(user)
    if prediction==[1]:
        prediction="No Cirrhosis"
    else:
        prediction="No Cirrhosis "
    messagebox.showinfo("Prediction", prediction)

# main application window
root = tk.Tk()
root.title("Cirrhosis Prediction System")

# a variable to hold the selected options
gender_var = tk.IntVar()
ascites_var = tk.IntVar()
drug_var = tk.IntVar()
hepatomegaly_var = tk.IntVar()
spiders_var = tk.IntVar()
edema_var = tk.IntVar()
status_var = tk.IntVar()
# radio buttons for selecting gender
male_radio = tk.Radiobutton(root, text="Male (0)", variable=gender_var, value=0, )
female_radio = tk.Radiobutton(root, text="Female (1)", variable=gender_var, value=1, )
# radio buttons for selecting ascites
no_ascites_radio = tk.Radiobutton(root, text="No Ascites (0)", variable=ascites_var, value=0, )
yes_ascites_radio = tk.Radiobutton(root, text="Ascites (1)", variable=ascites_var, value=1, )
# radio buttons for selecting drug
penicillamine_radio = tk.Radiobutton(root, text="D-penicillamine (0)", variable=drug_var, value=0, )
placebo_radio = tk.Radiobutton(root, text="Placebo (1)", variable=drug_var, value=1, )
# radio buttons for selecting hepatomegaly
no_hepatomegaly_radio = tk.Radiobutton(root, text="No Hepatomegaly (0)", variable=hepatomegaly_var, value=0, )
yes_hepatomegaly_radio = tk.Radiobutton(root, text="Hepatomegaly (1)", variable=hepatomegaly_var, value=1, )
# radio buttons for selecting spiders
no_spiders_radio = tk.Radiobutton(root, text="No Spiders (0)", variable=spiders_var, value=0, )
yes_spiders_radio = tk.Radiobutton(root, text="Spiders (1)", variable=spiders_var, value=1, )
# radio buttons for selecting edema
no_edema_radio = tk.Radiobutton(root, text="No Edema (0)", variable=edema_var, value=0, )
edema_radio = tk.Radiobutton(root, text="Edema (1)", variable=edema_var, value=1, )
resolved_edema_radio = tk.Radiobutton(root, text="Resolved Edema (-1)", variable=edema_var, value=-1, )
# Place the radio buttons in the window
tk.Label(root, text="Sex").grid(row=0,column=0)
male_radio.grid(row=0,column=1)
female_radio.grid(row=0,column=2)
# radio buttons for selecting ascites
tk.Label(root, text="Ascites").grid(row=1,column=0)
no_ascites_radio.grid(row=1,column=1)
yes_ascites_radio.grid(row=1,column=2)
# radio buttons for selecting penicillamine
tk.Label(root, text="Penicillamine").grid(row=2,column=0)
penicillamine_radio.grid(row=2,column=1)
placebo_radio.grid(row=2,column=2)
# radio buttons for selecting hepatomegaly
tk.Label(root, text="Hepatomegaly").grid(row=3,column=0)
no_hepatomegaly_radio.grid(row=3,column=1)
yes_hepatomegaly_radio.grid(row=3,column=2)
# radio buttons for selecting soiders
tk.Label(root, text="Spiders").grid(row=4,column=0)
no_spiders_radio.grid(row=4,column=1)
yes_spiders_radio.grid(row=4,column=2)
# radio buttons for selecting edema
tk.Label(root, text="Edema").grid(row=5,column=0)
no_edema_radio.grid(row=5,column=1)
edema_radio.grid(row=5,column=2)
resolved_edema_radio.grid(row=5,column=3)

entries = {}
fields=["Age (In Days)","Bilirubin [mg/dl]","Cholesterol [mg/dl]","Albumin [mg/dl]","Copper [ug/day]","Alk_Phos [U/liter]","SGOT [U/ml]","Tryglicerides [mg/dl]","Platelets [ml/1000]","Prothrombin [s]"]
i=6
for field in fields:
    tk.Label(root, text=field).grid(row=i,column=0)
    entry = tk.Entry(root)
    entry.grid(row=i,column=1)
    i+=1
    entries[field] = entry
    
submit_button = tk.Button(root, text="Predict", command=predict_user_data)
submit_button.grid(row=i+5, columnspan=6,)
# Start the GUI event loop
root.mainloop()

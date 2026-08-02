import pandas as pd

df = pd.read_excel("hourlyfootfall_till_current_date1.xlsx")
df.to_csv("hourlyfootfall_till_current_date1.csv", index=False)

print("CSV created successfully!")
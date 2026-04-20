# In[ ]:
import pandas as pd
df = pd.read_csv("online_retail_ll.csv", encoding='ISO-8859-1')
df.head()

# In[ ]:
df['return_risk'] = (df['Quantity'] < 0).astype(int)

# In[ ]:
df['return_risk'].value_counts()

# In[ ]:
df['return_risk'] = (df['Quantity'] < 0).astype(int)
df['return_risk'].value_counts()

# In[ ]:
return_rate = 22950 / (1044421 + 22950)

# In[ ]:
df.isnull().sum()

# In[ ]:
df = df.dropna(subset=['Customer ID'])
df['Description'] = df['Description'].fillna('Unknown')

# In[ ]:
df = df[df['Quantity'] != 0]

# In[ ]:
df = df[df['Price'] > 0]

# In[ ]:
df.isnull().sum()

# In[ ]:
df = df.dropna(subset=['Customer ID'])
df['Description'] = df['Description'].fillna('Unknown')
df = df[df['Quantity'] != 0]
df = df[df['Price'] > 0]

df.isnull().sum()

# In[ ]:
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# In[ ]:
df['year'] = df['InvoiceDate'].dt.year
df['month'] = df['InvoiceDate'].dt.month
df['day'] = df['InvoiceDate'].dt.day
df['weekday'] = df['InvoiceDate'].dt.weekday
df['hour'] = df['InvoiceDate'].dt.hour

# In[ ]:
df['total_price'] = df['Quantity'] * df['Price']

# In[ ]:
df['customer_total_orders'] = df.groupby('Customer ID')['Invoice'].transform('count')

# In[ ]:
df['product_return_rate'] = df.groupby('StockCode')['return_risk'].transform('mean')

# In[ ]:
df['customer_return_rate'] = df.groupby('Customer ID')['return_risk'].transform('mean')

# In[ ]:
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

df['year'] = df['InvoiceDate'].dt.year
df['month'] = df['InvoiceDate'].dt.month
df['day'] = df['InvoiceDate'].dt.day
df['weekday'] = df['InvoiceDate'].dt.weekday
df['hour'] = df['InvoiceDate'].dt.hour

df['total_price'] = df['Quantity'] * df['Price']

df['customer_total_orders'] = df.groupby('Customer ID')['Invoice'].transform('count')

df['product_return_rate'] = df.groupby('StockCode')['return_risk'].transform('mean')

df['customer_return_rate'] = df.groupby('Customer ID')['return_risk'].transform('mean')

# In[ ]:
df = df.sort_values(by=['Customer ID', 'InvoiceDate'])

# In[ ]:
features = [
    'Quantity',
    'Price',
    'total_price',
    'month',
    'day',
    'weekday',
    'hour',
    'product_return_rate',
    'customer_return_rate'
]

# In[ ]:
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# In[ ]:
sequence_length = 10

# In[ ]:
import numpy as np

X = []
y = []

for customer_id in df['Customer ID'].unique():
    customer_data = df[df['Customer ID'] == customer_id]
    
    for i in range(len(customer_data) - sequence_length):
        X.append(customer_data[features].iloc[i:i+sequence_length].values)
        y.append(customer_data['return_risk'].iloc[i+sequence_length])

X = np.array(X)
y = np.array(y)

# In[ ]:
print(X.shape)
print(y.shape)

# In[ ]:
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# In[ ]:
!pip install torch

# In[ ]:
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x[:, -1, :]   # last time step
        x = self.fc(x)
        return self.sigmoid(x)

# In[ ]:
import torch

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# In[ ]:
input_dim = X.shape[2]

model = TransformerModel(input_dim)

# In[ ]:
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# In[ ]:
epochs = 5
batch_size = 256

for epoch in range(epochs):
    model.train()
    
    for i in range(0, len(X_train_tensor), batch_size):
        X_batch = X_train_tensor[i:i+batch_size]
        y_batch = y_train_tensor[i:i+batch_size].unsqueeze(1)
        
        optimizer.zero_grad()
        
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# In[ ]:
model.eval()

with torch.no_grad():
    predictions = model(X_test_tensor).numpy()

# In[ ]:



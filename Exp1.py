import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing # to import colletion of data set 

housing=fetch_california_housing(as_frame=True) #to fetch the data set 
df=housing.frame

x=df["AveRooms"].values # to Extract X (features) and y (target)
y=df["MedHouseVal"].values  # .values converts the pandas series into a raw numpy array

np.random.seed(1) #to set a starting point 

indices=np.random.permutation(len(x)) # to rearrange order of incdices
split_limit=int(len(x)*0.2) # to set aside 20% for testing 

for_testing=indices[:split_limit] # dataset after the (colon) split for testing 
for_training=indices[split_limit:] # dataset before the (colon) split for testing

x_train,x_test=x[for_training],x[for_testing] # to slice the actual data 
y_train,y_test=y[for_training],y[for_testing] # to slice the actual data 

# Biased colume trick to consider slope and intercept as as one for easier calculatioins in future
ones_train = np.ones((x_train.shape[0], 1)) # Create an array of 1s with the exact same number of rows as our data
ones_test = np.ones((x_test.shape[0], 1))

x_train_b = np.c_[ones_train, x_train] #  to glue the column of 1s to the front of our data matrices
x_test_b = np.c_[ones_test, x_test]

def normal_equation(x, y):
    xt_dot_x = x.T.dot(x) # Step 4a: Calculate (X^T * X)
    inverse_part = np.linalg.inv(xt_dot_x)  # Step 4b: Calculate the inverse of that result
    xt_dot_y = x.T.dot(y) # Step 4c: Calculate (X^T * y)
    theta = inverse_part.dot(xt_dot_y) # Step 4d: Multiply the inverse part by the (X^T * y) part
    return theta
theta_ne = normal_equation(x_train_b, y_train) # to Run the function to get the analytical weights

def gradient_descent(x, y, alpha=0.001, iterations=5000):
    m = len(y)
    theta = np.zeros(x.shape[1]) # Initialize our weights vector to zeros [intercept=0, slope=0]
    
    for i in range(iterations):
        predictions = x.dot(theta)  # 1. Generate current predictions
        errors = predictions - y # 2. Calculate the difference (errors)
        gradient = (1 / m) * x.T.dot(errors) # 3. Compute the gradient (the direction of steepest error)
        theta = theta - alpha * gradient # 4. Take a small step downhill (subtract because we want lower error)
    return theta
theta_gd = gradient_descent(x_train_b, y_train, alpha=0.001, iterations=5000) # Run the loop to get the optimized weights

def evaluate_performance(y_true, y_pred):
    # Mean Squared Error formula
    mse = np.mean((y_true - y_pred) ** 2)
    
    # R-squared formula
    ss_residual = np.sum((y_true - y_pred) ** 2)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return mse, r2

y_pred_ne = x_test_b.dot(theta_ne)# Step 6a: Predict on the test set
y_pred_gd = x_test_b.dot(theta_gd)

mse_ne, r2_ne = evaluate_performance(y_test, y_pred_ne) # Step 6b: Calculate metrics
mse_gd, r2_gd = evaluate_performance(y_test, y_pred_gd)

print(f"Normal Equation -> MSE: {mse_ne:.4f}, R2: {r2_ne:.4f}") # Print results directly to console
print(f"Gradient Descent -> MSE: {mse_gd:.4f}, R2: {r2_gd:.4f}")

# Get the index order that would sort X_train from smallest to largest
sort_idx = np.argsort(x_train.flatten())

# Create a 1-row, 2-column plotting canvas
plt.figure(figsize=(14, 5))

# Subplot 1: Gradient Descent
plt.subplot(1, 2, 1)
plt.scatter(x_train, y_train, color='blue', alpha=0.1, label='Actual Data')
# Plot using the sorted indices to ensure a smooth, clean line
plt.plot(x_train[sort_idx], x_train_b[sort_idx].dot(theta_gd), color='red', linewidth=2, label='GD Fit')
plt.title(f"Gradient Descent\nMSE: {mse_gd:.4f} | R²: {r2_gd:.4f}")
plt.xlabel("AveRooms")
plt.ylabel("MedHouseVal")
plt.legend()

# Subplot 2: Normal Equation
plt.subplot(1, 2, 2)
plt.scatter(x_train, y_train, color='blue', alpha=0.1, label='Actual Data')
plt.plot(x_train[sort_idx], x_train_b[sort_idx].dot(theta_ne), color='green', linewidth=2, label='Normal Eq Fit')
plt.title(f"Normal Equation\nMSE: {mse_ne:.4f} | R²: {r2_ne:.4f}")
plt.xlabel("AveRooms")
plt.ylabel("MedHouseVal")
plt.legend()

plt.tight_layout()
plt.show()

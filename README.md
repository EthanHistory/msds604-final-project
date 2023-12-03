# msds604-final-project
Final project for time series

## How to add my model?
1. Add another python script in the root directory
2. In the script, define a class which inherits BaseModel in `base_model.py`
3. Implement `fit()` and `forecast()` method in the class
4. Look at how the demo model is doing cross-validation in `main.ipynb`
5. Create another cell in the notebook between `Data setup` and `Submission`.
6. in a similar way with the demo, write down the parameter sets and do cross-validation.
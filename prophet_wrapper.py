import pickle
import mlflow.pyfunc

class ProphetWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load the pickled Prophet model
        with open(context.artifacts["model"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input):
        # Expect input as a DataFrame with 'ds' column
        if "ds" not in model_input.columns:
            raise ValueError("Input DataFrame must contain a 'ds' column.")
        forecast = self.model.predict(model_input)
        return forecast[["ds", "yhat"]]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from sklearn.metrics import mean_squared_error


class KalmanFilter:
    def __init__(self, process_variance, measurement_variance):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 0.5

    def update(self, measurement):
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance
        blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate
        return self.posteri_estimate


class DataFilter:
    def __init__(self, file_path, column_name, alpha=0.3, kernel_size=31,
                 kalman_process_var=1e-4, kalman_measurement_var=1e-2):
        self.file_path = file_path
        self.column_name = column_name
        self.alpha = alpha
        self.kernel_size = kernel_size
        self.kalman = KalmanFilter(kalman_process_var, kalman_measurement_var)
        self.data = None
        self.original = []
        self.ema = []
        self.median = []
        self.kalman_filtered = []

    def load_data(self):
        df = pd.read_csv(self.file_path)
        self.data = df[self.column_name][:1500].values
        self.time = df['time'][:1500].values if 'time' in df.columns else np.arange(len(df))

        return df

    def apply_filters(self):
        buffer = np.zeros(self.kernel_size)
        ema_val = self.data[0]

        for i, val in enumerate(self.data):
            # EMA
            ema_val = self.alpha * val + (1 - self.alpha) * ema_val

            # Median
            buffer = np.roll(buffer, -1)
            buffer[-1] = val
            if i >= self.kernel_size - 1:
                med = medfilt(buffer, kernel_size=self.kernel_size)[self.kernel_size // 2]
            else:
                med = val

            # Kalman
            kalman_val = self.kalman.update(val)

            # Store
            self.original.append(val)
            self.ema.append(ema_val)
            self.median.append(med)
            self.kalman_filtered.append(kalman_val)
    
    def save_filtered_data(self, output_path="filtered_output.csv"):
        df_filtered = pd.DataFrame({
            "time": self.time,
            "original": self.original,
            "ema": self.ema,
            "median": self.median,
            "kalman": self.kalman_filtered
        })
        df_filtered.to_csv(output_path, index=False)
        print(f"Filtered data saved to: {output_path}")

    def compute_error_metrics(self):

        metrics = {}
        for label, filtered in {
            "ema": self.ema,
            "median": self.median,
            "kalman": self.kalman_filtered
        }.items():
            mse = mean_squared_error(self.original, filtered)
            rmse = np.sqrt(mse)
            metrics[label] = {"MSE": mse, "RMSE": rmse}

        print("Error Metrics (vs. Original):")
        for key, vals in metrics.items():
            print(f"  {key.upper()}: MSE = {vals['MSE']:.4f}, RMSE = {vals['RMSE']:.4f}")
        return metrics

    def plot(self, filters_to_plot=["original", "ema", "median", "kalman"], range_to_plot=None):
        # Dictionary to map names to the actual data
        filter_data = {
            "original": self.original,
            "ema": self.ema,
            "median": self.median,
            "kalman": self.kalman_filtered
        }

        # Labels and line styles
        styles = {
            "original": {"label": "Original", "style": "-", "alpha": 0.5},
            "ema": {"label": "EMA", "style": "--"},
            "median": {"label": "Median", "style": "-."},
            "kalman": {"label": "Kalman", "style": ":"}
        }

        # Determine range
        if range_to_plot is not None:
            start, end = range_to_plot
        else:
            start, end = 0, len(self.original)

        plt.figure(figsize=(12, 6))
        for key in filters_to_plot:
            if key in filter_data:
                plt.plot(
                    self.time[start:end],
                    filter_data[key][start:end],
                    linestyle=styles.get(key, {}).get("style", "-"),
                    alpha=styles.get(key, {}).get("alpha", 1.0),
                    label=styles.get(key, {}).get("label", key.capitalize())
                )
        plt.title(f"Filtered Plot: {', '.join([f.capitalize() for f in filters_to_plot])}")
        plt.xlabel("Time" if hasattr(self, "time") else "Sample Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()




if __name__ == "__main__":


 # Test data
    df_test = DataFilter("data_points_free_fall_40Hz - Copy.csv", column_name="xAccl", alpha=0.3, kernel_size=11,
                kalman_process_var=1e-5, kalman_measurement_var=1e-4) ##5-4:ok // 
    df_test.load_data()
    df_test.apply_filters()
    df_test.compute_error_metrics()
    df_test.save_filtered_data("filtered_free_fall_output.csv")
    # df_test.plot(filters_to_plot=["original", "median"])  
    # df_test.plot(filters_to_plot=["original", "ema"], range_to_plot=(1200, 1500))  
    df_test.plot(filters_to_plot=["original", "kalman"], range_to_plot=(1200, 1500))

    # Test data
    df_test = DataFilter("test_data.csv", column_name="theta", alpha=0.04, kernel_size=41) ##0.05
    df_test.load_data()
    df_test.apply_filters()
    df_test.compute_error_metrics()
    # df_test.save_filtered_data("filtered_test_output.csv")
    # df_test.plot(filters_to_plot=["original", "ema"])  
    # df_test.plot(filters_to_plot=["ema", "median"])
    # df_test.plot(filters_to_plot=["original", "kalman"])  

    # Sensor data
    # df_sensor = DataFilter("sensor_data.csv", column_name="xAccl", alpha=0.4, kernel_size=11)
    # df_sensor.load_data()
    # df_sensor.apply_filters()
    # df_sensor.save_filtered_data("filtered_sensor_output.csv")
    # df_sensor.plot(filters_to_plot=["original", "ema"])

    # df_data_3 = DataFilter("sensor_data_300ms.csv", column_name="xAccl", alpha=0.35, kernel_size=41)
    # df_data_3.load_data()
    # df_data_3.apply_filters()
    # df_data_3.plot(filters_to_plot=["original","ema"])

    # best_alpha = None
    # best_score = float('inf')
    # alpha_values = np.linspace(0.01, 0.99, 50)

    # results = []

    # for alpha in alpha_values:
    #     df_test = DataFilter("test_data.csv", column_name="theta", alpha=alpha, kernel_size=41)
    #     df_test.load_data()
    #     df_test.apply_filters()

    #     # Evaluate smoothness: variance of first difference (proxy for noise)
    #     derivative = np.diff(df_test.ema)
    #     smoothness = np.var(derivative)

    #     # Optional: also compare to original signal with MSE (not great if original is noisy)
    #     # mse = mean_squared_error(df_test.original, df_test.ema)

    #     results.append((alpha, smoothness))

    #     if smoothness < best_score:
    #         best_score = smoothness
    #         best_alpha = alpha

    # print(f"Best alpha (by smoothness): {best_alpha:.3f} with score {best_score:.6f}")

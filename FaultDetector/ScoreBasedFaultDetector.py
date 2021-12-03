from tqdm import tqdm

class ScoreBasedFaultDetector:

    def __init__(self, inference_result, threshold):
        self.inference_result = inference_result
        self.threshold = threshold

    def detect_faults(self):
        """
        For each entry of the inference result dataframe return whether a fault has been detected or not
        :return: a dictionary {image_name: fault_detected}
        """
        fault_detected = {}
        for index, row in tqdm(self.inference_result.iterrows(), total=len(self.inference_result)):
            fault_detected[index] = row.top_1 >= self.threshold[0] | row.top_1 <= self.threshold[1]

        return fault_detected

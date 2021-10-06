class MavFaultDetector:

    def __init__(self, inference_result, threshold):
        self.inference_result = inference_result
        self.threshold = threshold

    def detect_faults(self):
        """
        For each entry of the inference result dataframe return whether a fault has been detected or not
        :return: a dictionary {image_name: fault_detected}
        """
        # TODO: complete this function
        fault_detected = {}
        for index, row in self.inference_result.iterrows():
            fault_detected[index] = row.top_1 >= self.threshold

        return fault_detected

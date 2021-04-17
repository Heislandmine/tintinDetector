import tensorflow as tf


class Detector(object):
    def __init__(self, model_path) -> None:
        super().__init__()

        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()

    @property
    def input_width(self):
        return self.input_details["shape"][1]

    @property
    def input_height(self):
        return self.input_details["shape"][2]

    def run(self, input, th=0.1):
        input = self._preprocess(input)
        self.interpreter.set_tensor(0, input)
        self.interpreter.invoke()

        results = self._get_inference_results()

        return self._post_process(results, th)

    def _get_inference_results(self):

        results = []
        for output_detail in self.output_details:
            results.append(self.interpreter.get_tensor(output_detail["index"]))
        return results

    def _preprocess(self, input):
        img = input.resize((self.input_width, self.input_height))
        input_tensor = self.interpreter.get_tensor(0)
        input_tensor[:, :] = img
        input_tensor = input_tensor / 255.0
        return input_tensor

    def _post_process(self, results, th):
        boxes, _, scores, _ = results

        boxes = boxes[0][scores[0] > th]
        scores = scores[0][scores[0] > th]

        return boxes, scores

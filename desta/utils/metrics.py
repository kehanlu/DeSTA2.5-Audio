from whisper_normalizer.basic import BasicTextNormalizer

class ConsecutiveWordsAccuracyMetric(object):
    metric_name = "consecutive_words_accuracy"

    def __init__(self):
        self.normalizer = BasicTextNormalizer()
    
    def __call__(self, pred, label):
        """
        Compare consecutive strings in preds and labels.

        Input:
            pred: str
            label: str
        Output:
            accuracy: bool
        """

        pred = self.normalizer(pred)
        label = self.normalizer(label)

        return self.check_consecutive_words(long_string=pred, short_string=label)
    
    def check_consecutive_words(self, long_string, short_string):
        
        long_string_words = long_string.lower().split()
        short_string_words = short_string.lower().split()
        for i in range(len(long_string_words) - len(short_string_words) + 1):
            if long_string_words[i:i+len(short_string_words)] == short_string_words:
                return True
        return False
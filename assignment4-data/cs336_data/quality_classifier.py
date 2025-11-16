import fasttext
from typing import Optional, Dict

class QualityClassifier:
    def __init__(self, model_path: str, label_map: Optional[Dict[str, str]]):
        print(f"正在加载质量分类器模型: {model_path}")
        try:
            self.model = fasttext.load_model(model_path)
            print("模型加载完成。")
        except ValueError as e:
            print(f"模型加载失败，检查输入路径{model_path}是否正确")
            print(f"错误: {e}")
            raise
            
        if label_map is None:
            self.label_map = {
                '__label__cc': 'cc',
                '__label__wiki': 'wiki'
            }
        else:
            self.label_map = label_map

    def predict(self, text: str, notfound_label: str = 'unknown') -> tuple[str, float]:
        clean_text = text.replace('\n', ' ').replace('\r', ' ').strip()
        labels, scores = self.model.predict(clean_text, k=1)

        if not labels:
            return notfound_label, 0.0
        
        # 找到 'high_quality' 或 'low_quality' 的标签和分数
        original_label = labels[0]
        score = scores[0]

        mapped_label = self.label_map.get(original_label, original_label)

        return mapped_label, score

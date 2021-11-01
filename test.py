from roberta import Inference

model = Inference(model_path='model_1.pth')
encoded = model.predict('this is a test text')
print(encoded)

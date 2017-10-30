from drmodel import DrModel

m = DrModel()
m.load_model("drmodel")
m.save_activation_map("preds")
m.save_saliency_map("preds")

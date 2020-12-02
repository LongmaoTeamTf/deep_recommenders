"""
@Description: 训练模型
@version: 1.0.0
@License: MIT
@Author: Wang Yao
@Date: 2020-12-01 16:26:40
@LastEditors: Wang Yao
@LastEditTime: 2020-12-02 14:29:01
"""
import argparse
import yaml
from deep_recommend.recommend.ctr.embedding_mlp import EmbeddingMLP
from deep_recommend.recommend.ctr.trainner import CtrModelTrainer
from deep_recommend.recommend.ctr.deepfm.fm import FmPart
from deep_recommend.recommend.ctr.dcn.cross_net import CrossNet
from deep_recommend.recommend.ctr.xdeepfm.xdeepfm import CIN


parser = argparse.ArgumentParser()
parser.add_argument("mode", type=str, choices=["ctr"])
parser.add_argument("model", type=str, choices=["deepfm", "dcn", "xdeepfm"])
parser.add_argument("dataset", type=str, choices=["criteo"])
parser.add_argument("train_data_dir", type=str)
parser.add_argument("valid_data_dir", type=str)
parser.add_argument("test_data_dir", type=str)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--epochs", type=int, default=10)


ctr_models_yaml = "deep_recommend/recommend/ctr/ctr_models.yml"


def run_ctr_model(args):
    """ 训练模型 """
    configs = yaml.load(open(ctr_models_yaml), Loader=yaml.FullLoader)
    model_yaml = configs.get("models").get(args.model).get("yaml")
    model_config = yaml.load(open(model_yaml), Loader=yaml.FullLoader)
    dataset_yaml = configs.get("datasets").get(args.dataset).get("yaml")
    dataset_config = yaml.load(open(dataset_yaml), Loader=yaml.FullLoader)

    embedding_mlp = EmbeddingMLP(
        dataset_config.get("dense_features"),
        dataset_config.get("sparse_features"),
        model_config.get("ff").get("hidden_sizes").split(","),
        model_config.get("ff").get("hidden_activation"),
        model_config.get("ff").get("hidden_dropout_rates").split(","),
        model_config.get("logits").get("size"),
        model_config.get("logits").get("activation"),
        model_config.get("model").get("name"),
        model_config.get("model").get("loss"),
        model_config.get("model").get("optimizer")
    )
    if args.model == "deepfm":
        explicit_part = FmPart(model_config.get("fm").get("factors"))
    elif args.model == "dcn":
        explicit_part = CrossNet(model_config.get("dcn").get("cross_layers_num"))
    elif args.model == "xdeepfm":
        explicit_part = CIN(
            model_config.get("cin").get("feature_maps").split(","),
            model_config.get("cin").get("feature_embedding_dim"))
    else:
        return f"Unsupport model {args.model}"

    model = embedding_mlp(explicit_part)

    trainer = CtrModelTrainer(
        model, 
        args.dataset, 
        batch_size=args.batch_size, 
        epochs=args.epochs)

    trainer(
        args.train_data_dir, 
        args.valid_data_dir, 
        args.test_data_dir)


if __name__ == "__main__":
    args = parser.parse_args()
    print("Model: {}".format(args.model))
    print("Dataset: {}".format(args.dataset))
    print("Train: batch_size={}, epochs={}".format(args.batch_size, args.epochs))
    print("Train Data Dir: {}".format(args.train_data_dir))
    print("Valid Data Dir: {}".format(args.valid_data_dir))
    print("Test Data Dir: {}".format(args.test_data_dir))
    run_ctr_model(args)

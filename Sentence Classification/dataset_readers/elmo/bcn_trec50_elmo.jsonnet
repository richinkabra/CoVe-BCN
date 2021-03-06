{
  // Slightly modified version of the bi-attentative classification model with ELMo
  // from "Deep contextualized word representations" (https://www.aclweb.org/anthology/N18-1202),
  // trained on 5-class Stanford Sentiment Treebank.
  // There is a trained model available at https://allennlp.s3.amazonaws.com/models/sst-5-elmo-biattentive-classification-network-2018.09.04.tar.gz
  // with test accuracy of 54.7%.
  "dataset_reader":{
    "type": "trec_50",
    "use_subtrees": true,
    "granularity": "5-class",
    "token_indexers": {
      "tokens": {
        "type": "single_id"
      },
      "elmo": {
        "type": "elmo_characters"
      }
    }
  },
  "validation_dataset_reader":{
    "type": "trec_50",
    "use_subtrees": false,
    "granularity": "5-class",
    "token_indexers": {
      "tokens": {
        "type": "single_id"
      },
      # If using ELMo in the BCN, add an elmo_characters
      # token_indexers.
      "elmo": {
        "type": "elmo_characters"
      }
    }
  },
  "train_data_path": "<path to training data>",
  "validation_data_path": "<path to test data>",
  "test_data_path": "<path to test data>",
  "model": {
    "type": "bcn",
    # The BCN model will consume the arrays generated by the ELMo token_indexer
    # independently of the text_field_embedder, so we do not include the elmo key here.
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz",
            "type": "embedding",
            "embedding_dim": 300,
            "trainable": false
        }
      }
    },
    "embedding_dropout": 0.5,
    "pre_encode_feedforward": {
        "input_dim": 1324,
        "num_layers": 1,
        "hidden_dims": [300],
        "activations": ["relu"],
        "dropout": [0.25]
    },
    "encoder": {
      "type": "lstm",
      "input_size": 300,
      "hidden_size": 300,
      "num_layers": 1,
      "bidirectional": true
    },
    "integrator": {
      "type": "lstm",
      "input_size": 1800,
      "hidden_size": 300,
      "num_layers": 1,
      "bidirectional": true
    },
    "integrator_dropout": 0.1,
    "elmo": {
      "options_file": "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
      "weight_file": "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
      "do_layer_norm": false,
      "dropout": 0.0,
      "num_output_representations": 1
    },
    "use_input_elmo": true,
    "use_integrator_output_elmo": false,
    "output_layer": {
        "input_dim": 2400,
        "num_layers": 3,
        "output_dims": [1200, 600, 6],
        "pool_sizes": 4,
        "dropout": [0.2, 0.3, 0.0]
    }
  },
  "iterator": {
      "type": "basic",
      "batch_size" : 32
  },
  "trainer": {
    "num_epochs": 40,
    "patience": 5,
    "grad_norm": 5.0,
    "validation_metric": "+accuracy",
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    }
  }
}
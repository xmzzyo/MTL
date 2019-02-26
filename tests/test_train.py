# -*- coding: utf-8 -*-
import logging
import os
import re

import torch
from allennlp.commands.evaluate import evaluate
from allennlp.commands.train import create_serialization_dir, datasets_from_params, logger
from allennlp.common import Params
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.util import prepare_environment, prepare_global_logging, get_frozen_and_tunable_parameter_names, \
    dump_metrics
from allennlp.data import Vocabulary, DataIterator, DatasetReader
from allennlp.data.dataset import Batch
from allennlp.models import Model, archive_model
from allennlp.training import Trainer

dep_file_path = 'dep/data/PMT-train.conllu'
srl_file_path = 'srl/data/PMT-srl-train.conllu'

dep_train_data = 'dep/data/PMT-train.conllu'
dep_valid_data = 'dep/data/PMT-valid.conllu'

srl_train_data = ''
srl_valid_data = ''

# dep_instances = UniversalDependenciesDatasetReader().read(dep_file_path)
# # srl_instances = UniversalDependenciesDatasetReader().read(srl_file_path)
# dep_vocab = Vocabulary.from_instances(dep_instances)
# # srl_vocab = Vocabulary.from_instances(srl_instances)
#
# dep_model = BiaffineDependencyParser.from_params(vocab=dep_vocab, params=model_params)
# # srl_model = BiaffineDependencyParser.from_params(vocab=srl_vocab, params=model_params)
#
# dep_optimizer = torch.optim.SGD(dep_model.parameters(), 0.01, momentum=0.9)
# # srl_optimizer = DenseSparseAdam(betas=[0.9, 0.9])
#
# dep_iterator = DataIterator.from_params(iterator_params)
# dep_iterator.index_with(dep_vocab)
#
# params = Params.from_file('dep/config.json')
# dep_trainer = Trainer(dep_model, dep_file_path, dep_iterator, dep_train_data, dep_valid_data)


param_file = 'dep/config.json'
params = Params.from_file(param_file)

CONFIG_NAME = 'config.json'
DEFAULT_WEIGHTS = 'weight.th'


def train_model(params: Params,
                serialization_dir: str,
                file_friendly_logging: bool = False,
                recover: bool = False,
                force: bool = False) -> Model:
    prepare_environment(params)

    create_serialization_dir(params, serialization_dir, recover, force)
    prepare_global_logging(serialization_dir, file_friendly_logging)

    cuda_device = params.params.get('trainer').get('cuda_device', -1)
    if isinstance(cuda_device, list):
        for device in cuda_device:
            check_for_gpu(device)
    else:
        check_for_gpu(cuda_device)

    params.to_file(os.path.join(serialization_dir, CONFIG_NAME))

    all_datasets = datasets_from_params(params)
    datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", all_datasets))

    for dataset in datasets_for_vocab_creation:
        if dataset not in all_datasets:
            raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {dataset}")

    logger.info("From dataset instances, %s will be considered for vocabulary creation.",
                ", ".join(datasets_for_vocab_creation))

    if recover and os.path.exists(os.path.join(serialization_dir, "vocabulary")):
        vocab = Vocabulary.from_files(os.path.join(serialization_dir, "vocabulary"))
    else:
        vocab = Vocabulary.from_params(
            params.pop("vocabulary", {}),
            (instance for key, dataset in all_datasets.items()
             for instance in dataset
             if key in datasets_for_vocab_creation)
        )

    model = Model.from_params(vocab=vocab, params=params.pop('model'))

    # Initializing the model can have side effect of expanding the vocabulary
    vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

    iterator = DataIterator.from_params(params.pop("iterator"))
    iterator.index_with(vocab)
    validation_iterator_params = params.pop("validation_iterator", None)
    if validation_iterator_params:
        validation_iterator = DataIterator.from_params(validation_iterator_params)
        validation_iterator.index_with(vocab)
    else:
        validation_iterator = None

    train_data = all_datasets['train']
    validation_data = all_datasets.get('validation')
    test_data = all_datasets.get('test')

    trainer_params = params.pop("trainer")
    no_grad_regexes = trainer_params.pop("no_grad", ())
    for name, parameter in model.named_parameters():
        if any(re.search(regex, name) for regex in no_grad_regexes):
            parameter.requires_grad_(False)

    frozen_parameter_names, tunable_parameter_names = \
        get_frozen_and_tunable_parameter_names(model)
    logger.info("Following parameters are Frozen  (without gradient):")
    for name in frozen_parameter_names:
        logger.info(name)
    logger.info("Following parameters are Tunable (with gradient):")
    for name in tunable_parameter_names:
        logger.info(name)

    trainer_choice = trainer_params.pop_choice("type",
                                               Trainer.list_available(),
                                               default_to_first_choice=True)
    trainer = Trainer.by_name(trainer_choice).from_params(model=model,
                                                          serialization_dir=serialization_dir,
                                                          iterator=iterator,
                                                          train_data=train_data,
                                                          validation_data=validation_data,
                                                          params=trainer_params,
                                                          validation_iterator=validation_iterator)

    evaluate_on_test = params.pop_bool("evaluate_on_test", False)
    params.assert_empty('base train command')

    try:
        metrics = trainer.train()
    except KeyboardInterrupt:
        # if we have completed an epoch, try to create a model archive.
        if os.path.exists(os.path.join(serialization_dir, DEFAULT_WEIGHTS)):
            logging.info("Training interrupted by the user. Attempting to create "
                         "a model archive using the current best epoch weights.")
            archive_model(serialization_dir, files_to_archive=params.files_to_archive)
        raise

    # Now tar up results
    archive_model(serialization_dir, files_to_archive=params.files_to_archive)

    logger.info("Loading the best epoch weights.")
    best_model_state_path = os.path.join(serialization_dir, 'best.th')
    best_model_state = torch.load(best_model_state_path)
    best_model = model
    best_model.load_state_dict(best_model_state)

    if test_data and evaluate_on_test:
        logger.info("The model will be evaluated using the best epoch weights.")
        test_metrics = evaluate(
            best_model, test_data, validation_iterator or iterator,
            cuda_device=trainer._cuda_devices[0]  # pylint: disable=protected-access
        )
        for key, value in test_metrics.items():
            metrics["test_" + key] = value

    elif test_data:
        logger.info("To evaluate on the test set after training, pass the "
                    "'evaluate_on_test' flag, or use the 'allennlp evaluate' command.")

    dump_metrics(os.path.join(serialization_dir, "metrics.json"), metrics, log=True)

    return best_model


train_model(params, 'output')

exit()

reader = DatasetReader.from_params(params['dataset_reader'])
# The dataset reader might be lazy, but a lazy list here breaks some of our tests.
train_instances = list(reader.read(params['train_data_path']))
valid_instances = list(reader.read(params['validation_data_path']))

# Use parameters for vocabulary if they are present in the config file, so that choices like
# "non_padded_namespaces", "min_count" etc. can be set if needed.
if 'vocabulary' in params:
    vocab_params = params['vocabulary']
    vocab = Vocabulary.from_params(params=vocab_params, instances=train_instances.extend(valid_instances))
else:
    vocab = Vocabulary.from_instances(train_instances.extend(valid_instances))
dep_model = Model.from_params(vocab=vocab, params=params['model'])
print(dep_model)
iterator = DataIterator.from_params(params.pop("iterator"))
iterator.index_with(vocab)
train_dataset = Batch(train_instances)
train_dataset.index_instances(vocab)

valid_dataset = Batch(valid_instances)
valid_dataset.index_instances(vocab)

# dep_trainer = Trainer(dep_model, dep_file_path, dep_iterator, dep_train_data, dep_valid_data)
trainer_params = params.pop("trainer")

trainer = Trainer.from_params(model=dep_model,
                              serialization_dir='',
                              iterator=iterator,
                              train_data=train_dataset,
                              validation_data=valid_dataset,
                              params=trainer_params,
                              validation_iterator=iterator)

metrics = trainer.train()

archive_model('data/output')

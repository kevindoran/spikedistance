"""
This file aims to be a project-independent training-loop and associated
scaffolding.
"""

from contextlib import contextmanager
import logging
import math
import pathlib
from typing import Any, Callable, List, Optional, Tuple, TypeAlias, Union
import numpy as np
import torch
import plotly
import plotly.graph_objects as go
import retinapy._logging
import retinapy.models

_logger = logging.getLogger(__name__)

"""
Max training time before a new recovery is made.

Standard checkpointing only happens every epoch. But a checkpoint for recovery
purposes will be made every 30 minutes. Only a single recovery is kept.
"""
RECOVERY_CKPT_PERIOD_SEC = 30 * 60

"""
Here we take part in the rite of of passage for a deep learning project by
yet again reinventing the training loop architecture. No one wants their 
project stitched together with the callbacks of some soon to be abandonded or 
rewritten DL framework.
"""


class Trainable:
    """Encapsulates a dataset, model input-output and loss function.

    This class is needed in order to be able to train multiple models and
    configurations with the same training function. The training function
    is too general to know about how to route the data into and out of a model,
    evaluate the model or how to take a model output and create a prediction.

    Redesign from function parameters to a class
    --------------------------------------------
    The class began as a dumb grouping of the parameters to the train function:
    train_ds, val_ds, test_ds, model, loss_fn, forward_fn, val_fn, and more—there
    were so many parameters that they were grouped together into a NamedTuple.
    However, functions like forward_fn and val_fn would need to be given
    the model, loss_fn and forward_fn in order to operate. This is the exact
    encapsulation behaviour that classes achieve, so the NamedTuple was made
    into a class. Leaning into the use of classes, forward_fn and val_fn were
    made into methods of the class, while the rest became properties. This
    change is noteworthy as customizing the forward or validation functions
    now requires defining a new class, rather than simply passing in a new
    function. Although, nothing is stopping someone from defining a class that
    simply wraps a function and passes the arguments through.

    Flexibility to vary the data format
    -----------------------------------
    Why things belongs inside or outside this class can be understood by
    realizing that the nature of the datasets are known here. As such,
    any function that needs to extract the individual parts of a dataset
    sample will need to know what is in each sample. Such a function is a
    good candidate to appear in this class.

    While in some cases you can separate the datasets from the models, this
    isn't always easy or a good idea. A model for ImageNet can easily be
    separated from the dataset, as the inputs and outputs are so standard; but
    for the spike prediction, the model output is quite variable. Consider the
    distance array model which outputs an array, whereas a Poisson-distribution
    model will output a single number. Training is done with these outputs, and
    involves the dataset producing sample tuples that have appropriate elements
    (distance fields, for example). The actual inference is an additional
    calculation using these outputs.

    In other words, the procedure of taking the model input and model output
    from a dataset sample and feeding it to the model, then calculating the
    loss and doing the inference—none of these steps can be abstracted to be
    ignorant of either the model or the dataset.

    Handles dataloader creation
    ---------------------------
    train.py handled this responsibility for a number of months; however,
    it became a problem once I needed more control over the evaluation
    function. For evaluation, I wanted to run the slow inference proceedure
    on a subset of the cells and additionally create input-output videos
    for them. This was specific to just one of the trainables. The ideal
    approach would be for the evaluation function to get or create a number
    of extra data loaders (one for each cluster) and then run the more detailed
    evaluation on each. There wasn't a good way to do this without moving
    the full responsibility of dataloader handing into the Trainable class.
    The current approach is to use an intermediate (DataManager) that lazily
    creates datasets; this allows trainables to create multiple variations of
    a dataset as needed, without them being created up front.

    Other libraries
    ---------------
    Compared to Keras and FastAI: Trainable encapsulates a lot less than
    Keras's Model or FastAI's Learner.

    At this point, I'm not eager to use the FastAI API, as I don't
    want to discover later that it's too limiting in some certain way. It's
    quite possible that it's already too prescriptive. Reading the docs, it's
    not clear what parts of Learner's internals are exposed for customization.
    If all "_" prefixed methods are not meant to be customized, then it's
    already too restrictive. Notably, there seems to be an expected format for
    the elements of the dataset, which I want to avoid. The reason for this is
    that the distance arrays are intermediate results, and while I want to
    train on them, I would like to evaluate based on quick approximate
    inference and make predictions using much more expensive and accurate 
    inference routines. So the data doesn't fall nicely into (X,y) type data,
    and the metrics are not consistent across training and evaluation.

    In addition, at least at the momemt, FastAI's library provides a lot more
    abstraction/generalization than I need, which can make it harder for
    myself (or others) to understand what is going on. This might end up being
    a mistake, as the growing code might reveal itself to provide abstraction 
    boundaries that are already handled nicely in FastAI.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        label: str,
        # TODO: manage GPU usage.
    ):
        """
        Args:
            train_ds: the dataset to train the model on.
            val_ds: the dataset to evaluate the model on and guide model
                training. This dataset is used to decide what model states to
                keep, and when to stop training, if early termination is
                enabled.
            test_ds: the test dataset. Similar to the validation dataset, this
                dataset is available for evaluating the model; however,
                its purpose is to be a datasat which has no influence
                on guiding the training. This includes any hyperparameter
                turing and the design of inference procedures. If more stages
                of data holdout are desired, then the validation dataset
                should be split again, rather than using the test dataset.
            model: the PyTorch model to train.
            label: a string label for this trainable.
        """
        self.model = model
        self.label = label

    # I think I want to make these functions, with the semantics that they
    # may be created on the fly on each call.
    @property
    def train_ds(self):
        raise NotImplementedError("Override")

    @property
    def val_ds(self):
        raise NotImplementedError("Override")

    @property
    def test_ds(self):
        raise NotImplementedError("Override")

    def forward(self, sample):
        """Run the model forward.

        Args:
            sample: a single draw from the train or validation data loader.

        Returns:
            (output, loss): the model output and the loss, as a tuple.
        """
        raise NotImplementedError("Override")

    """
    Some considerations about the evaluate functions.
       - The evaluation functions should be given the power and responsibility
         to create the datasets. This is because they may want to change the
         nature of a dataset. For example, the spike prediction models want
         to change ds stride and to filter in/out certain clusters.
       - The evaluation functions should not need to care about how to create
         a dataloader from a dataset. Things like worker counts and batch
         sizes should not need to be handled.
       - There should two separate routines: one for the training dataset and
         one for the validation dataset. The reason for the distinction 
         will mostly often be the sizes of the datasets. 
       - It might be desirable to have a third function that will be run
         at the end of training, or outside of training, and is expected to be 
         a long-running. 
    """

    def evaluate_train(self, dl_fn):
        """Run the evaluation procedure on the train dataset.

        Args:
            dl_fn: a function to convert the train dataset into a dataloader.

        Returns:
            metrics: a str:float dictionary containing evaluation metrics. It
                is expected that this dictionary at least contains 'loss' and
                'accuracy' metrics.
        """
        raise NotImplementedError("Override")

    def evaluate_val(self, dl_fn):
        """Run the evaluation procedure on the val dataset.

        Args:
            dl_fn: a function to convert the train dataset into a dataloader.

        Returns:
            metrics: a str:float dictionary containing evaluation metrics. It
                is expected that this dictionary at least contains 'loss' and
                'accuracy' metrics.
        """
        raise NotImplementedError("Override")

    def evaluate_full(self, ds, dl_fn):
        """Run a pontentially long evaluation procedure on the given dataset.

        This isn't implemented anywhere yet. The idea here is to carve out
        some space for running a longer evaluation procedure on a dataset. This
        function is not called during the training routine.

        Args:
            ds: the dataset to evaluate on.
            dl_fn: a function to convert the train dataset into a dataloader.
        """
        raise NotImplementedError("Override")

    def in_device(self):
        """Returns the device on which the model expects input to be located.

        Most likely, the whole model is on a single device, and it is
        sufficient to use `next(self.model.parameters()).device`.
        """
        raise NotImplementedError("Override")

    def __str__(self) -> str:
        return f"Trainable ({self.label})"

    def model_summary(self, batch_size: int) -> str:
        """Returns a detailed description of the model.

        Args:
            batch_size: the batch size to use when creating the model summary.

        At the moment, this is called by train() with the intent of saving
        out a file containing info like torchinfo.summary. The torch module
        input shape isn't known by train(), so the actual summary creation
        must be done somewhere like Trainable.

        Override this to add more features.
        """
        return f"Trainable ({self.label})"


class DataManager:
    """
    Create train, val & test Pytorch Datasets.

    The train, val & test datasets must be created almost identically, except
    for small changes to options like shuffle.

    The reason for choosing factory functions to be passed around instead of
    the actual datasets is that there are cases where we need to create
    more, and the initial train, val, test datasets are not sufficient.
    Alternatively, we want access to dataloaders and not datasets or vise-versa
    The case that initiated this refactor was the need to split a validation
    set into multiple 1-cluster datasets for the purpose of evaluating
    autoregressive inference.

    Another benefit is that the datasets aren't created eagerly, which can
    save memory if they aren't needed, which is typically the case for the
    test dataset while training.

    train.py will construct the DatasetManager and give it to the Trainable.
    The DatasetManager will be held by the Trainable, and so train.py will
    no longer need to construct and pass around dataloaders.
    """

    def train_ds(self) -> torch.utils.data.Dataset:
        raise NotImplementedError()

    def val_ds(self) -> torch.utils.data.Dataset:
        raise NotImplementedError()

    def test_ds(self) -> torch.utils.data.Dataset:
        raise NotImplementedError()


def _create_dataloaders(batch_size, eval_batch_size, num_workers, pin_memory):
    """
    Create dataloaders for train & val datasets.

    What is returned is a pair of factory functions λ.ds → dl.

    The test dataset is not included because it is not used during training.
    So far, the test_ds has never been needed during training, so it has
    not been included here.

    History note: it use to be the case that the dataloaders were created
    here and returned. However, this was changed for a number of reasons.
    Firstly, having dataset creation be lazy saves memory, and so dataloader
    creation should also be lazy. Secondly, we want to give more control to
    the Trainable, and instead of passing in a dataloader, we want to just
    say "please evaluate using the train dataset". We don't need to give in
    the dataset, as the Trainable is where train.py got access to it anyway.
    This might suggest a function like `trainable.evaluate_train(self)`;
    however, one missing piece is that the Trainable doesn't know how to
    create a dataloader. A few options remain:
        1. We give this responsibility to the DataManager. The downside of this
        is that the DataManager starts becoming a bit omniscient, and creating
        one becomes a bit of an unnecessary pain when we don't intend to train
        and may only want to run some inference.
        2. We could do 1. above, but leave a lot of default initialized
        options that we would expect train.py to set, such as batch_size.
        3. Pass a factory function λ.ds → dl to the Trainable when we call
        `trainable.evaluate_train(self)`. This is the approach taken here. We
        leave train.py with the responsibility of creating the dataloaders,
        and we keep the DataManager as a simple factory for datasets.
        3. We could pass both a dataset and a dataloader or a dataset and a
        factory function λ.ds → d. This wont work, as in many cases the creation
        of the dataset must be done by the evaluate() function. For example,
        setting the stride of the dataset.
    """
    # Setting pin_memory=True. This is generally recommended when training on
    # Nvidia GPUs. See:
    #   - https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
    #   - https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
    train_dl_fn = lambda ds: torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dl_fn = lambda ds: torch.utils.data.DataLoader(
        ds,
        batch_size=eval_batch_size,
        # For debugging, it's nice to see a variety:
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_dl_fn, val_dl_fn


@contextmanager
def evaluating(model):
    """
    Context manager to set the model to eval mode and then back to train mode.

    Used this to prevent an exception leading to unexpected training state.
    """
    original_mode = model.training
    model.eval()
    try:
        model.eval()
        yield
    finally:
        # Switch back to the original training mode.
        model.train(original_mode)


class TrainCallbackList:
    """
    Document the available training callbacks.
    """

    def on_batch_start(self, step: int):
        pass

    def on_batch_end(self, step: int):
        pass

    def on_eval_start(self, step: int):
        pass

    def on_eval_end(self, step: int):
        pass


class Callback:
    """
    Currently, training is done in a stateless manner in train().

    An alternative is to have a stateful training setup, by creating a
    class Trainer that keeps data such as step and optimizer as properties.
    The benefit of a stateful approach is that callbacks can have a reference
    to the trainer and all callbacks can have an empty signature. The downside
    is that the callbacks are now coupled to the trainer, and so it's harder
    to reuse them in other contexts. It's clearly far more powerful to have
    access to a Trainer, and I'm leaning towards this approach. For now,
    we will be satisfied with a very basic callback interface without access
    to a stateful trainer.
    """

    def before_train(self, trainable: Trainable, step: int):
        """
        Called after train() has completed setup, but before the first batch.

        Setup includes things like creating any loggers and configuring
        the optimizer.

        Step is 0 unless training is being resumed. Although, there currently
        isn't any support for resuming with an initialized step.
        """
        pass

    def before_batch(self, trainable: Trainable, step: int):
        pass

    def after_batch(self, trainable: Trainable, step: int):
        pass

    def after_train(self, trainable: Trainable, step: int):
        pass

    def before_eval(self, trainable: Trainable, step: int):
        pass

    def after_eval(self, trainable: Trainable, step: int):
        pass


class TensorLogger(Callback):
    "TODO: add options for switching between means and histograms." ""

    tag_class = retinapy.models.TensorTag

    def __init__(self, tb_logger, steps_til_log=50):
        self.model = None
        self.step = 0
        self._is_enabled = False
        self.tb_logger = tb_logger
        self.last_log = 0
        self.steps_til_log = steps_til_log
        self._hooks = []

    def before_train(self, trainable: Trainable, step: int):
        self.model = trainable.model
        self._add_hook(self.model)

    def before_batch(self, trainable: Trainable, step: int):
        self.step = step
        self._is_enabled = self.step - self.last_log > self.steps_til_log

    def after_batch(self, trainable: Trainable, step: int):
        self.step = step
        if self._is_enabled:
            self.last_log = self.step
        self._is_enabled = False

    def before_eval(self, trainable: Trainable, step: int):
        self._is_enabled = False

    def after_eval(self, trainable: Trainable, step: int):
        self._is_enabled = True

    def _log(self, value, label):
        if self._is_enabled:
            self.tb_logger.add_histogram(label, value, self.step)

    @torch.no_grad()
    def _on_forward(self, module, args, kwargs, output) -> None:
        """
        Return None when the output is not modified
        """
        # args[0] is the passthrough tensor, x (same as output).
        label = kwargs.get("label", args[1]) if len(args) > 1 else None
        no_label_arg = label is None or label == ""
        if no_label_arg:
            assert hasattr(module, "label"), "TensorTag must have a label."
            label = module.label
        self._log(output, label)

    def _add_hook(self, model):
        def _add_hook(submodule):
            if isinstance(submodule, self.tag_class):
                self._hooks.append(
                    submodule.register_forward_hook(
                        self._on_forward, with_kwargs=True
                    )
                )

        model.apply(_add_hook)

    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()

    def __del__(self):
        self._remove_hooks()


class TrainingTimers:
    """Collect timers here for convenience."""

    def __init__(self):
        self.batch = retinapy._logging.Timer()
        self.epoch = retinapy._logging.Timer()
        self.validation = retinapy._logging.Timer()
        self.recovery = retinapy._logging.Timer()

    @staticmethod
    def create_and_start():
        timer = TrainingTimers()
        timer.batch.restart()
        timer.epoch.restart()
        timer.recovery.restart()
        return timer


class EarlyStopping:
    def __init__(self, min_epochs, patience):
        self.patience = patience
        self.min_epochs = min_epochs
        self.best_loss = None
        self.best_epoch = 0
        self.cur_epoch = 0

    def update(self, loss):
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = self.cur_epoch
            do_early_stop = False
        else:
            is_warmup = self.cur_epoch < self.min_epochs
            is_within_patience = (
                self.cur_epoch - self.best_epoch <= self.patience
            )
            do_early_stop = not (is_warmup or is_within_patience)
        return do_early_stop


def configure_cuda():
    # Enable Cuda convolution auto-tuning. For more details, see:
    #   https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    torch.backends.cudnn.benchmark = True

    # Create a GradScalar. For more details, see:
    #   https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
    grad_scaler = torch.cuda.amp.GradScaler()
    return grad_scaler


def create_optimizer(model_params, lr, weight_decay):
    """
    No need to abstract the optimizer creation into the interface yet; however,
    it's used in two places now (train & lr_find), so at least deduplicate it
    to avoid insidious bugs.
    """
    res = torch.optim.AdamW(
        # Why use lr with AdamW, if we use a scheduler?
        # Because the scheduler isn't used for the first epoch? So start
        # slow, lr=lr/25.
        model_params,
        lr=lr,
        # Default eps is 1e-8, which can give us gradients in excess of 1e8.
        # Setting it lower. Inspired by Jeremy Howard.
        eps=1e-5,
        weight_decay=weight_decay,
        # Beta default is (0.9, 0.999). Here, we instead follow Jeremy Howard
        # into the dark world of hyperparameter tuning and use (0.9, 0.99).
        # See: https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html
        betas=(0.9, 0.99),
    )
    return res


def lr_sweep(
    trainable: Trainable,
    lr_min: float,
    lr_max: float,
    num_lr_steps: int,
    weight_decay: float,
    batch_size: int,
    num_workers: int = 4,
    divergence_threshold: float = 5.0,
) -> Tuple[List[float], List[float]]:
    """Find a good learning rate.

    Args:
        trainable: The trainable to use for training.
        lr_min: The minimum learning rate to try.
        lr_max: The maximum learning rate to try.
        num_lr_steps: The number of learning rates to try.
        weight_decay: The weight decay to use.
        batch_size: The batch size to use.
        num_workers: The number of workers to use for the dataloaders.
        divergence_threshold: The threshold for divergence. If the loss
            increases above threshold*best_smoothed_loss, then we stop.
    """
    # Exponential schedule between lr_min and lr_max.
    sched = (
        lr_min * (lr_max / lr_min) ** torch.linspace(0, 1, num_lr_steps)
    ).tolist()

    # Note: the lr finder currently doesn't use the grad scalar. An issue?
    configure_cuda()
    model = trainable.model
    model.cuda()
    model.train()
    optimizer = create_optimizer(model.parameters(), lr_min, weight_decay)

    def _set_lr(l):
        # Torch's optimizers store parameter references in a list of dicts.
        #  [ {"params": [p1, p2, ...], "lr": l},
        #    {"params": [p1, p2, ...]}, ... ]
        # Any option such as "lr" that isn't specified in the dict is
        # inherited from the default value set in the optimizer's constructor.
        # It's interesting that these parameter groups are identified by
        # their index in the list, not by any string label.
        for p in optimizer.param_groups:
            # No need to override the default, as parameter groups have the
            # default value set in their individual lr field.
            p["lr"] = l

    train_dl_fn, _ = _create_dataloaders(
        batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
    )
    train_dl = train_dl_fn(trainable.train_ds)
    num_epochs = math.ceil(num_lr_steps / len(train_dl))

    losses = []
    loss_meter = retinapy._logging.MovingAverageMeter(beta=0.1)
    train_step = 0
    best_loss = float("inf")
    for epoch in range(num_epochs):
        for sample in train_dl:
            # Set learning rate
            _set_lr(sched[train_step])
            # Do the zero-forward-backward-step and turn around.
            optimizer.zero_grad(set_to_none=True)
            _, loss = trainable.forward(sample)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            loss_meter.update(loss.item())
            # Update best loss.
            if loss_meter.avg < best_loss:
                best_loss = loss_meter.avg
            else:
                if loss_meter.avg > divergence_threshold * best_loss:
                    break
            train_step += 1
            if train_step >= num_lr_steps:
                break
    res = (sched[: len(losses)], losses)
    return res


def longest_valley_lr(lr_sched, losses):
    if len(lr_sched) != len(losses):
        raise ValueError(
            "lr_sched and losses must have the same length."
            f"Got ({len(lr_sched)}, {len(losses)})"
        )
    if len(lr_sched) == 0:
        raise ValueError("lr_sched and losses must have at least one element.")
    L = len(losses)
    # Record left-side valley lengths. A valley at position i either continues
    # any previous valley, making its length extend, or it starts a new valley
    # of length 1.
    valleys = np.ones(shape=L, dtype=int)
    for i in range(1, L):
        for j in range(i):
            if losses[i] < losses[j]:
                # This is really weird. Wonder what they were doing.
                # It's more like a tally.
                valleys[i] = max(valleys[j] + 1, valleys[i])
    valley_end = np.argmax(valleys)
    # +1, otherwise we will go past the beginning of the array. Needed anyway
    # as the length is end inclusive.
    valley_start = valley_end - valleys[valley_end] + 1
    # Take a point β of the way through the valley, in log scale.
    β = 0.3
    lr_idx = round(valley_start + β * (valley_end - valley_start))
    assert valley_start <= lr_idx <= valley_end
    res = lr_sched[lr_idx]
    return res, (valley_start, valley_end, lr_idx)


def plot_lr_sweep(lr_sched, losses, valley_start_idx, valley_end_idx, lr_idx):
    # Two subplots, one ontop of the other.
    fig = plotly.subplots.make_subplots(
        rows=2, cols=1, subplot_titles=["Learning rate schedule", "Losses"]
    )
    fig.append_trace(
        go.Scatter(x=np.arange(len(lr_sched)), y=lr_sched), row=1, col=1
    )

    def accumulate(v, current=[0]):
        β = 0.95
        current[0] = β * current[0] + (1 - β) * v
        return current[0]

    loss_smoothed = [accumulate(v) for v in losses]

    fig.append_trace(
        go.Scatter(x=lr_sched, y=losses, name="loss"), row=2, col=1
    )
    fig.append_trace(
        go.Scatter(x=lr_sched, y=loss_smoothed, name="loss (smooth)"),
        row=2,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            x=[lr_sched[valley_start_idx]],
            y=[losses[valley_start_idx]],
            marker=dict(color="green", size=10),
            name="valley start",
        ),
        row=2,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            x=[lr_sched[valley_end_idx]],
            y=[losses[valley_end_idx]],
            marker=dict(color="red", size=10),
            name="valley end",
        ),
        row=2,
        col=1,
    )
    fig.append_trace(
        go.Scatter(
            x=[lr_sched[lr_idx]],
            y=[losses[lr_idx]],
            marker=dict(color="blue", size=10),
            name="suggested lr",
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(title="Step", row=1, col=1)
    fig.update_yaxes(title="Learning rate", row=1, col=1)
    fig.update_xaxes(title="Learning rate", type="log", row=2, col=1)
    fig.update_yaxes(title="Loss", row=2, col=1)
    return fig


def suggest_lr(
    trainable: Trainable,
    lr_min: float,
    lr_max: float,
    num_lr_steps: int,
    weight_decay: float,
    batch_size: int,
    num_workers: Optional[int] = None,
    divergence_threshold: Optional[float] = None,
) -> Tuple[float, plotly.graph_objects.Figure]:
    sweep_args = [
        trainable,
        lr_min,
        lr_max,
        num_lr_steps,
        weight_decay,
        batch_size,
        num_workers,
    ]
    if divergence_threshold is not None:
        sweep_args.append(divergence_threshold)
    lr_sched, losses = lr_sweep(*sweep_args)
    suggested_lr, (
        valley_start,
        valley_end,
        lr_idx,
    ) = longest_valley_lr(lr_sched, losses)
    fig = plot_lr_sweep(
        lr_sched,
        losses,
        valley_start,
        valley_end,
        lr_idx,
    )
    logging.info(
        f"Suggested LR: {suggested_lr} for trainable " f'"{trainable.label}"'
    )
    return suggested_lr, fig


def inspect_scheduler(
    scheduler_fn: Callable[
        [torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler
    ],
    num_steps: int,
) -> Tuple[float, float, plotly.graph_objects.Figure]:
    dummy_parameters = torch.nn.Parameter(torch.zeros(1))
    dummy_optm = torch.optim.AdamW([dummy_parameters], lr=1e-3)
    scheduler = scheduler_fn(dummy_optm)
    lrs = []
    for i in range(num_steps):
        # Some schedulers complain if optimizer.step() is not called before the
        # scheduler step.
        dummy_optm.step()
        scheduler.step()
        lr = dummy_optm.param_groups[0]["lr"]
        lr2 = scheduler.get_last_lr()[0]
        assert lr == lr2
        lrs.append(lr)
    fig = go.Figure(
        go.Scatter(
            x=np.arange(num_steps),
            y=lrs,
        )
    )
    min_lr = min(lrs)
    max_lr = max(lrs)
    last_lr = lrs[-1]
    return min_lr, max_lr, last_lr, fig


SchedulerFn: TypeAlias = Callable[
    [torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler
]


def train(
    trainable: Trainable,
    num_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    out_dir: Union[str, pathlib.Path],
    steps_til_log: int = 1000,
    steps_til_eval: Optional[int] = None,
    evals_til_eval_train_ds: Optional[int] = None,
    early_stopping: Optional[EarlyStopping] = None,
    initial_checkpoint: Optional[Union[str, pathlib.Path]] = None,
    logger: Optional[Any] = None,
    num_workers: int = 4,
    pin_memory: bool = False,
    log_activations: bool = False,
    scheduler_fn: Optional[SchedulerFn] = None,
    callbacks: Optional[List[Callback]] = None,
    eval_batch_size: Optional[int] = None,
):
    """
    Train a model.

    This is a training loop that works with any Trainable object.

    It encapsulates basic functionality like logging, checkpointing and
    choosing when to run an evalutaion. Users might be just as well off
    by copying the code to use as a baseline and modifying it to their needs.
    """
    logging.info(f"Training {trainable.label}")
    out_dir = pathlib.Path(out_dir)
    # Setup output (logging & checkpoints).
    if not logger:
        tensorboard_dir = out_dir / "tensorboard"
        logger = retinapy._logging.TbLogger(tensorboard_dir)

    eval_batch_size = eval_batch_size or batch_size

    if callbacks is None:
        callbacks = []
    if log_activations:
        callbacks.append(TensorLogger(logger.writer))

    # Load the model & loss fn.
    # The order here is important when resuming from checkpoints. We must:
    # 1. Create model & log structure
    #     - logging the model summary has a secondary function of testing the
    #       model integrity: it will fail if the model cannot be constructed.
    #       We want to do this test before the time consuming step of loading
    #       the dataset.
    # 2. Send model to target device
    # 3. Load datasets
    # 4. Create optimizer & scheduler
    #     - these can be present in checkpoints.
    #     - the dataset length must be known to construct the initial scheduler.
    # 5. Initialize from checkpoint
    #     - last, after we have the model, optimizer and scheduler created.
    #
    # Further details:
    # Another reason the order is crucial is that the optimizer must be on the
    # gpu before having it's parameters populated, as there is no optimizer.gpu()
    # method (possibly coming: https://github.com/pytorch/pytorch/issues/41839).
    # An alternative would be to use the map_location argument. See the
    # discussion: https://github.com/pytorch/pytorch/issues/2830.
    model = trainable.model
    model.cuda()
    grad_scaler = configure_cuda()
    optimizer = torch.optim.AdamW(
        # Why use lr with AdamW, if we use a scheduler?
        # Because the scheduler isn't used for the first epoch? So start
        # slow, lr=lr/25.
        model.parameters(),
        lr=lr / 25,
        weight_decay=weight_decay,
        # Beta default is (0.9, 0.999). Here, we instead follow Jeremy Howard
        # into the dark world of hyperparameter tuning and use (0.9, 0.99).
        # See: https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html
        betas=(0.9, 0.99),
    )

    model.train()
    # Before going any further, log model structure.
    out_file = out_dir / "model_summary.txt"
    with open(out_file, "w") as f:
        # This is allowed to fail, as if the model has issues, we want to
        # get the errors from the actual training forward pass.
        summary = None
        try:
            summary = trainable.model_summary(batch_size=batch_size)
        except Exception as e:
            msg = (
                "Failed to generate model summary. Exception raised:\n"
                f"{str(e)}"
            )
            _logger.error(msg)
            summary = msg
        f.write(str(summary))

    metric_tracker = retinapy._logging.MetricTracker(out_dir)

    # Load the data.
    train_dl_fn, val_dl_fn = _create_dataloaders(
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    # Go ahead and create the train_dl.
    train_dl = train_dl_fn(trainable.train_ds)
    total_num_steps = num_epochs * len(train_dl)

    # Scheduler is down here, after the dataset is loaded (to check size).
    split_percent = np.array(
        [
            len(trainable.train_ds),
            len(trainable.val_ds),
            len(trainable.test_ds),
        ]
    )
    split_percent = np.round(100 * split_percent / split_percent.sum(), 1)
    # Log dataset details.
    _logger.info(
        "Dataset lengths:\n"
        f"\t{'train:':<6} {len(trainable.train_ds):,}\n"
        f"\t{'val:':<6} {len(trainable.val_ds):,}\n"
        f"\t{'test:':<6} {len(trainable.test_ds):,}\n"
        f"Split ratio: {split_percent}"
    )
    # Log train loop details.
    _logger.info(
        "Train loop:\n"
        f"\tbatch size: {batch_size}\n"
        f"\tepochs: {num_epochs}\n"
        f"\tsteps per epoch: {len(train_dl):,}\n"
        f"\ttotal steps: {total_num_steps:,}"
    )

    def _scheduler_fn(_optimizer):
        return torch.optim.lr_scheduler.OneCycleLR(
            _optimizer,
            max_lr=lr,
            # Either specify steps & epochs, or specify total steps manually.
            # steps_per_epoch=len(train_dl),
            # epochs=num_epochs,
            total_steps=total_num_steps,
            # Testing
            three_phase=True,
        )

    scheduler_fn = _scheduler_fn if scheduler_fn is None else scheduler_fn
    scheduler = scheduler_fn(optimizer)
    lr_min, lr_max, lr_last, sched_fig = inspect_scheduler(
        scheduler_fn, total_num_steps
    )
    plotly.io.write_image(sched_fig, str(out_dir / "lr_schedule.png"))
    logger.log_plotly("lr_schedule", 0, [sched_fig], log_group="train")
    _logger.info(
        f"Learning rate schedule:\n"
        f"\tscheduler: {scheduler.__class__.__name__}\n"
        f"\tmin lr: {lr_min}\n"
        f"\tmax lr: {lr_max}"
        f"\tlast lr: {lr_last}"
    )
    # Record how many times schedule step is skipped due to grad scalar backoff.
    # We want to track this as too many steps will cause the resulting lr
    # schedule clipping to be significant.
    num_lr_steps_skipped = 0

    model_saver = retinapy._logging.ModelSaver(
        out_dir, trainable.model, optimizer, scheduler
    )

    if initial_checkpoint is not None:
        retinapy.models.load_model_and_optimizer(
            model, initial_checkpoint, optimizer, scheduler
        )

    def _eval(use_train_ds: bool = False):
        for cb in callbacks:
            cb.before_eval(trainable, step)
        label = "train-ds" if use_train_ds else "val-ds"
        _logger.info(f"Running evaluation {label}")
        with evaluating(model), torch.no_grad(), timers.validation:
            if use_train_ds:
                eval_results = trainable.evaluate_train(train_dl_fn)
            else:
                eval_results = trainable.evaluate_val(val_dl_fn)
        logger.log(step, eval_results, label)
        if "metrics" not in eval_results:
            raise ValueError("Trainable.evaluate() must return metrics.")
        metrics = eval_results["metrics"]
        assert (
            metrics[0].name == "loss"
        ), "Currently, by convention, the first metric must be loss."
        retinapy._logging.print_metrics(metrics)
        logger.log_scalar(step, "eval-time", timers.validation.elapsed(), label)
        _logger.info(
            f"Finished evaluation in {round(timers.validation.elapsed())} sec "
            f"(rolling ave: {round(timers.validation.rolling_duration())} sec)"
        )
        for cb in callbacks:
            cb.after_eval(trainable, step)
        return metrics

    _logger.info("Starting training loop.")
    step = 0
    num_evals = 0
    timers = TrainingTimers.create_and_start()
    for cb in callbacks:
        cb.before_train(trainable, step)
    # Do an initial eval.
    _eval()
    # The length of the epoch inner loop is now clearly too long to be able
    # to understand it at a glance.
    for epoch in range(num_epochs):
        timers.epoch.restart()
        # Meters to calculate smooth values for terminal logging.
        beta = max(1e-3, 1 / steps_til_log)
        loss_meter = retinapy._logging.MovingAverageMeter(beta, name="loss")
        lr_meter = retinapy._logging.MovingAverageMeter(beta, name="lr")
        for batch_step, sample in enumerate(train_dl):
            for cb in callbacks:
                cb.before_batch(trainable, step)
            timers.batch.restart()
            # set_to_none=True is suggested to improve performance, according to:
            #   https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
            # A recipe for autocast, grad scaling and set_to_none:
            #   https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                model_out, total_loss = trainable.forward(sample)
            grad_scaler.scale(total_loss).backward()
            grad_scaler.step(optimizer)
            # Recording the before and after scale. Why? The grad scaler can
            # create NaN gradients in the first few iterations, and when it
            # does, it skips the updating of the optimizer. We want to
            # know when this happens so that we can also skip the learning
            # rate scheduler update.
            # ref: https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930/10
            scale_before = grad_scaler.get_scale()
            grad_scaler.update()
            assert grad_scaler.get_backoff_factor() < 1.0, (
                "The logic for skipping the learning rate scheduler "
                "relies on the backoff factor being less than 1.0, which "
                "it is the default (0.5)."
            )
            scale_has_decreased = scale_before > grad_scaler.get_scale()
            skip_lr_sched = scale_has_decreased
            # get_last_lr() returns a list, but we only have one param group.
            last_lr = scheduler.get_last_lr()[0]
            lr_meter.update(last_lr)

            metrics = [
                retinapy._logging.Metric("loss", total_loss.item()),
            ]
            logger.log_metrics(step, metrics, log_group="train")
            logger.log_scalar(step, "epoch", epoch, log_group="train")
            logger.log_scalar(step, "lr", last_lr, log_group="train")

            # We log total_loss directly to logging framework, but log a
            # smoothed loss to the console.
            loss_meter.update(total_loss.item())
            # Log to console.
            if step % steps_til_log == 0:
                model_mean = torch.mean(model_out)
                model_sd = torch.std(model_out)
                elapsed_min, elapsed_sec = divmod(
                    round(timers.epoch.elapsed()), 60
                )
                # Floor total minutes to align with batch minutes.
                total_hrs, total_min = divmod(
                    int(timers.epoch.total_elapsed() / 60), 60
                )
                _logger.info(
                    f"ep: {epoch}/{num_epochs} | "
                    f"step:{batch_step:>4}/{len(train_dl)} "
                    f"({batch_step/len(train_dl):>3.0%}) "
                    f"{round(1/timers.batch.rolling_duration()):>2}/s | "
                    f"⏲  {elapsed_min:>1}m:{elapsed_sec:02d}s "
                    f"({total_hrs:>1}h:{total_min:02d}m) | "
                    f"loss: {loss_meter.avg:.3f} | "
                    f"lr: {lr_meter.avg:.2e} | "
                    f"out μ (σ): {model_mean:>3.2f} ({model_sd:>3.2f})"
                )

            # Evaluate.
            # (step + 1), as we don't want to evaluate on the first step.
            if steps_til_eval and (batch_step + 1) % steps_til_eval == 0:
                is_near_epoch_end = batch_step + steps_til_eval >= len(train_dl)
                if not is_near_epoch_end:
                    _eval()
                    num_evals += 1
                    if (
                        evals_til_eval_train_ds
                        and num_evals % evals_til_eval_train_ds == 0
                    ):
                        _eval(use_train_ds=True)
            if not skip_lr_sched:
                scheduler.step()
            else:
                num_lr_steps_skipped += 1
                _logger.info(
                    "Skipping lr scheduler step due to grad scaler backoff. "
                    f"Total skipped steps: {num_lr_steps_skipped} "
                    f"({round(float(num_lr_steps_skipped)/(step+1)):>3.0%})"
                )
            step += 1
            # Recovery.
            # Don't allow training to proceed too long without checkpointing.
            if timers.recovery.elapsed() > RECOVERY_CKPT_PERIOD_SEC:
                model_saver.save_recovery()
                timers.recovery.restart()
            for cb in callbacks:
                cb.after_batch(trainable, step)
            # One more batch to dust!

        _logger.info(
            f"Finished epoch in {round(timers.epoch.elapsed())} secs "
            f"(rolling duration: "
            f"{round(timers.epoch.rolling_duration())} s/epoch)"
        )
        # Evaluate and save at end of epoch.
        metrics = _eval()
        num_evals += 1
        if evals_til_eval_train_ds and num_evals % evals_til_eval_train_ds == 0:
            _eval(use_train_ds=True)

        assert (
            metrics[0].name == "loss"
        ), "By convention, the first metric must be loss."
        # If this on_metric_end type of behaviour grows, consider switching
        # to callbacks.
        improved_metrics = metric_tracker.on_epoch_end(metrics, epoch)
        model_saver.save_checkpoint(epoch, improved_metrics)
        timers.recovery.restart()
        if early_stopping:
            should_stop = early_stopping.update(metrics[0])
            if should_stop:
                _logger.info(
                    "Early stopping triggered (patience: "
                    f"{early_stopping.patience} epochs)."
                )
                break
    for cb in callbacks:
        cb.after_train(trainable, step)
    _logger.info(
        f"Finished training. {round(timers.epoch.total_elapsed())} "
        "secs elsapsed."
    )

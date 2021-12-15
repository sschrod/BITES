import numpy as np
import pandas as pd
import torch
from bites.utils.Simple_Network import *
from bites.utils.loss import cox_ph_loss
from torch import Tensor
from torch import nn


class DeepSurv_Base(nn.Module):
    duration_col = 'duration'
    event_col = 'event'
    treatment_col = 'treatment'

    def reset(self):
        self.baseline_hazards_ = []
        self.baseline_cumulative_hazards_ = []

    def _compute_baseline_hazards(self, input, df_target, max_duration, eval_=True, num_workers=0):
        if max_duration is None:
            max_duration = np.inf

        pred= self.predict_numpy(input)

        # Here we are computing when expg when there are no events.
        #   Could be made faster, by only computing when there are events.
        return (df_target
                .assign(expg=np.exp(pred))
                .groupby(self.duration_col)
                .agg({'expg': 'sum', self.event_col: 'sum'})
                .sort_index(ascending=False)
                .assign(expg=lambda x: x['expg'].cumsum())
                .pipe(lambda x: x[self.event_col] / x['expg'])
                .fillna(0.)
                .iloc[::-1]
                .loc[lambda x: x.index <= max_duration]
                .rename('baseline_hazards'))

    def target_to_df(self, target):
        durations, events, treatment = target
        df = pd.DataFrame({self.duration_col: durations, self.event_col: events, self.treatment_col: treatment})
        return df

    def compute_baseline_hazards(self, input=None, target=None, max_duration=None, sample=None,
                                 set_hazards=True):
        """Computes the Breslow estimates form the data defined by `input` and `target`
        (if `None` use training data).
        Typically call
        model.compute_baseline_hazards() after fitting.

        Keyword Arguments:
            input  -- Input data (train input) (default: {None})
            target  -- Target data (train target) (default: {None})
            max_duration {float} -- Don't compute estimates for duration higher (default: {None})
            sample {float or int} -- Compute estimates of subsample of data (default: {None})
            batch_size {int} -- Batch size (default: {8224})
            set_hazards {bool} -- Set hazards in model object, or just return hazards. (default: {True})

        Returns:
            pd.Series -- Pandas series with baseline hazards. Index is duration_col.
        """
        if (input is None) and (target is None):
            if not hasattr(self, 'training_data'):
                raise ValueError("Need to give a 'input' and 'target' to this function.")
            input, target = self.training_data
        df = self.target_to_df(target)  # .sort_values(self.duration_col)
        if sample is not None:
            if sample >= 1:
                df = df.sample(n=sample)
            else:
                df = df.sample(frac=sample)

        if self.treatment is not None:
            mask=df[self.treatment_col]==self.treatment
            input=input[mask]
            df = df[mask]
        else:
            print('self.treatment not set, Baseline Hazards are computed for all given samples!!')

        base_haz = self._compute_baseline_hazards(input, df, max_duration)
        if set_hazards:
            self.compute_baseline_cumulative_hazards(set_hazards=True, baseline_hazards_=base_haz)
        return base_haz

    def compute_baseline_cumulative_hazards(self, input=None, target=None, max_duration=None, sample=None,
                                            batch_size=8224, set_hazards=True, baseline_hazards_=None,
                                            eval_=True, num_workers=0):
        """See `compute_baseline_hazards. This is the cumulative version."""
        if ((input is not None) or (target is not None)) and (baseline_hazards_ is not None):
            raise ValueError("'input', 'target' and 'baseline_hazards_' can not both be different from 'None'.")
        if baseline_hazards_ is None:
            baseline_hazards_ = self.compute_baseline_hazards(input, target, max_duration, sample, batch_size,
                                                              set_hazards=False, eval_=eval_, num_workers=num_workers)
        assert baseline_hazards_.index.is_monotonic_increasing, \
            'Need index of baseline_hazards_ to be monotonic increasing, as it represents time.'
        bch = (baseline_hazards_
               .cumsum()
               .rename('baseline_cumulative_hazards'))
        if set_hazards:
            self.baseline_hazards_=baseline_hazards_
            self.baseline_cumulative_hazards_=bch
        return bch

    def predict_cumulative_hazards(self, input, treatment, max_duration=None, batch_size=8224, verbose=False,
                                   baseline_hazards_=None, eval_=True, num_workers=0):
        """See `predict_survival_function`."""
        if type(input) is pd.DataFrame:
            input = self.df_to_input(input)
        if baseline_hazards_ is None:
            if not hasattr(self, 'baseline_cumulative_hazards_'):
                raise ValueError('Need to compute baseline_hazards_. E.g run `model.compute_baseline_hazards()`')
            baseline_hazards_ = self.baseline_cumulative_hazards_

        assert baseline_hazards_.index.is_monotonic_increasing,\
            'Need index of baseline_hazards_ to be monotonic increasing, as it represents time.'
        return self._predict_cumulative_hazards(input,treatment, max_duration, batch_size, verbose, baseline_hazards_,
                                                eval_, num_workers=num_workers)

    def _predict_cumulative_hazards(self, input, treatment, max_duration, batch_size, verbose, baseline_hazards_,
                                    eval_=True, num_workers=0):
        max_duration = np.inf if max_duration is None else max_duration
        if baseline_hazards_ is self.baseline_hazards_:
            bch = self.baseline_cumulative_hazards_
        else:
            bch = self.compute_baseline_cumulative_hazards(set_hazards=False,
                                                           baseline_hazards_=baseline_hazards_)
        bch = bch.loc[lambda x: x.index <= max_duration]
        pred = self.predict_numpy(input)
        expg = np.exp(pred).reshape(1, -1)
        return pd.DataFrame(bch.values.reshape(-1, 1).dot(expg),
                            index=bch.index)

    def predict_surv_df(self, input, treatment=None, max_duration=None, batch_size=8224, verbose=False,
                        baseline_hazards_=None,
                        eval_=True, num_workers=0):
        """Predict survival function for `input`. S(x, t) = exp(-H(x, t))
        Require computed baseline hazards.
        Arguments:
            input {np.array, tensor or tuple} -- Input x passed to net.
        Keyword Arguments:
            max_duration {float} -- Don't compute estimates for duration higher (default: {None})
            batch_size {int} -- Batch size (default: {8224})
            baseline_hazards_ {pd.Series} -- Baseline hazards. If `None` used `model.baseline_hazards_` (default: {None})
            eval_ {bool} -- If 'True', use 'eval' mode on net. (default: {True})
            num_workers {int} -- Number of workers in created dataloader (default: {0})
        Returns:
            pd.DataFrame -- Survival estimates. One columns for each individual.
        """
        return np.exp(-self.predict_cumulative_hazards(input, max_duration, batch_size, verbose, baseline_hazards_))

    def predict_surv(self, input, treatment, max_duration=None, batch_size=8224, numpy=None, verbose=False,
                     baseline_hazards_=None, eval_=True, num_workers=0):
        """Predict survival function for `input`. S(x, t) = exp(-H(x, t))
        Require compueted baseline hazards.
        Arguments:
            input {np.array, tensor or tuple} -- Input x passed to net.
        Keyword Arguments:
            max_duration {float} -- Don't compute estimates for duration higher (default: {None})
            batch_size {int} -- Batch size (default: {8224})
            numpy {bool} -- 'False' gives tensor, 'True' gives numpy, and None give same as input
                (default: {None})
            baseline_hazards_ {pd.Series} -- Baseline hazards. If `None` used `model.baseline_hazards_` (default: {None})
            eval_ {bool} -- If 'True', use 'eval' mode on net. (default: {True})
            num_workers {int} -- Number of workers in created dataloader (default: {0})
        Returns:
            pd.DataFrame -- Survival estimates. One columns for each individual.
        """
        surv = self.predict_surv_df(input, treatment, max_duration, batch_size, verbose, baseline_hazards_,
                                        eval_, num_workers)

        return surv.values.transpose()





class DeepSurv_Loss(torch.nn.Module):
    """Loss function for the Tar net structure
    uses the same Cox Loss as PyCox but separates between different treatment classes
    """

    def __init__(self,*_):
        super().__init__()

    def forward(self, log_h: Tensor, durations: Tensor, events: Tensor,*_) -> Tensor:
        return cox_ph_loss(log_h, durations, events)


class DeepSurv(DeepSurv_Base):
    """Network structure similar to the DeepHit paper, but without the residual
    connections (for simplicity).
    """
    def __init__(self, in_features, num_nodes, out_features=1, batch_norm=True, dropout=None):
        super().__init__()
        self.shared_net = MLPVanilla(
            in_features, num_nodes, out_features,
            batch_norm, dropout,
        )
        self.baseline_hazards_ = []
        self.baseline_cumulative_hazards_ = []
        self.treatment=[]

    def forward(self, input, *_):

        log_h = self.shared_net(input)
        return log_h

    def predict(self, X):
        self.eval()
        out = self(X)
        self.train()
        return out

    def predict_numpy(self, X):
        self.eval()
        X = torch.Tensor(X)
        out = self(X)
        self.train()
        return out.detach()

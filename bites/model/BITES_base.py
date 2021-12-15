import numpy as np
import pandas as pd
import torch
from bites.utils.Simple_Network import *
from bites.utils.loss import cox_ph_loss
from geomloss import SamplesLoss
from torch import Tensor
from torch import nn


class BITES_Base(nn.Module):
    duration_col = 'duration'
    event_col = 'event'
    treatment_col = 'treatment'

    def reset(self):
        self.baseline_hazards_ = []
        self.baseline_cumulative_hazards_ = []

    def _compute_baseline_hazards(self, input, df_target, max_duration):
        if max_duration is None:
            max_duration = np.inf

        pred,_ = self.predict_numpy(input, df_target['treatment'].to_numpy())

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
        """Computes the Breslow estimates form the data defined by `input` and `target` for a binary treatment choice!
        Each of the outcome arms is associated with  its own hazard function based on the respective samples.
        (if `None` use training data).
        Typically call
        model.compute_baseline_hazards() after fitting.

        Keyword Arguments:
            input  -- Input data (train input) (default: {None})
            target  -- Target data (train target) (default: {None}) <- Holds treatment choices!!
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

        if not hasattr(df, 'treatment'):
            raise ValueError("target hast to be of the form (duration, event, treatment)! Treatment is missing!")
        mask0 = df['treatment'] == 0
        mask1 = df['treatment'] == 1

        df0 = df[mask0]
        df1 = df[mask1]

        X0 = input[df0.index.values]
        X1 = input[df1.index.values]

        base_haz0 = self._compute_baseline_hazards(X0, df0, max_duration)
        base_haz1 = self._compute_baseline_hazards(X1, df1, max_duration)

        if set_hazards:
            self.compute_baseline_cumulative_hazards(set_hazards=True, baseline_hazards_=base_haz0)
            self.compute_baseline_cumulative_hazards(set_hazards=True, baseline_hazards_=base_haz1)
        return base_haz0, base_haz1

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
            self.baseline_hazards_.append(baseline_hazards_)
            self.baseline_cumulative_hazards_.append(bch)
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

        out = []
        for i, hazards in enumerate(baseline_hazards_):
            assert hazards.index.is_monotonic_increasing, 'Need index of baseline_hazards_ to be monotonic increasing, as it represents time.'
            inp = input[treatment == i]
            t = treatment[treatment == i]
            out.append(self._predict_cumulative_hazards(inp, t, max_duration, batch_size, verbose, hazards,
                                                        eval_, num_workers=num_workers))
        return out

    def _predict_cumulative_hazards(self, input, treatment, max_duration, batch_size, verbose, baseline_hazards_,
                                    eval_=True, num_workers=0):
        max_duration = np.inf if max_duration is None else max_duration
        """if baseline_hazards_ is self.baseline_hazards_:
            bch = self.baseline_cumulative_hazards_
        else:
            bch = self.compute_baseline_cumulative_hazards(set_hazards=False,
                                                           baseline_hazards_=baseline_hazards_)"""
        bch = baseline_hazards_
        bch = bch.loc[lambda x: x.index <= max_duration]
        pred, _ = self.predict_numpy(input, treatment)

        expg = np.exp(pred).reshape(1, -1)
        return pd.DataFrame(bch.values.reshape(-1, 1).dot(expg),
                            index=bch.index)

    def predict_surv_df(self, input, treatment, max_duration=None, batch_size=8224, verbose=False,
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
        hazards_all = self.predict_cumulative_hazards(input, treatment, max_duration, batch_size, verbose,
                                                      baseline_hazards_,
                                                      eval_, num_workers)
        out = []
        for hazards in hazards_all:
            out.append(np.exp(-hazards))
        return out

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
        surv_all = self.predict_surv_df(input, treatment, max_duration, batch_size, verbose, baseline_hazards_,
                                        eval_, num_workers)
        out = []
        for surv in surv_all:
            out.append(surv.values.transpose())
        return out

    def predict_surv_counterfactual(self, input, treatment, max_duration=None, batch_size=8224, numpy=None,
                                    verbose=False,
                                    baseline_hazards_=None, eval_=True, num_workers=0):
        """!New function to predict counterfactuals! So far only usable for two treatments!
        Requires compueted baseline hazards.
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
                list of np.arrays
                list[0] -- counterfactual survial estimates for treatment 0
                list[1] -- counterfactual survial estimates for treatment 1
            """
        if not len(np.unique(treatment)) == 2:
            raise ValueError('Needs exactly 2 different treatment choices (more treatments must be analysed manually)')
        treatment_cf = 1 - treatment
        surv_all = self.predict_surv_df(input, treatment_cf, max_duration, batch_size, verbose, baseline_hazards_,
                                        eval_, num_workers)
        out = []
        for surv in surv_all:
            out.append(surv.values.transpose())
        out.reverse()
        return out

    def predict_surv_counterfactual_df(self, input, treatment, max_duration=None, batch_size=8224, numpy=None,
                                       verbose=False,
                                       baseline_hazards_=None, eval_=True, num_workers=0):
        """!New function to predict counterfactuals! So far only usable for two treatments!
        Requires compueted baseline hazards.
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
                list of np.arrays
                list[0] -- counterfactual survial estimates for treatment 0
                list[1] -- counterfactual survial estimates for treatment 1
            """
        if not len(np.unique(treatment)) == 2:
            raise ValueError('Needs exactly 2 different treatment choices (more treatments must be analysed manually)')
        treatment_cf = 1 - treatment
        hazards_all = self.predict_cumulative_hazards(input, treatment_cf, max_duration, batch_size, verbose,
                                                      baseline_hazards_,
                                                      eval_, num_workers)
        out = []
        for hazards in hazards_all:
            out.append(np.exp(-hazards))
        out.reverse()
        return out



class BITES_Loss(torch.nn.Module):
    """Loss function for the Tar net structure
    uses the same Cox Loss as PyCox but separates between different treatment classes
    """

    def __init__(self, alpha=0, blur=0.05):
        self.alpha = alpha
        self.blur = blur
        super().__init__()

    def forward(self, log_h: Tensor, out: Tensor, durations: Tensor, events: Tensor, treatments: Tensor) -> Tensor:
        mask0 = treatments == 0
        mask1 = treatments == 1

        loss_t0 = cox_ph_loss(log_h[mask0], durations[mask0], events[mask0])
        loss_t1 = cox_ph_loss(log_h[mask1], durations[mask1], events[mask1])

        """Imbalance loss"""
        # p=torch.sum(mask1)/(torch.sum(mask0)+torch.sum(mask0))
        # sig = torch.tensor(self.sig)
        p = torch.sum(mask1) / treatments.shape[0]
        # imbalace_loss = mmd2_rbf(out, treatments, p, sig)
        # imbalace_loss = mmd2_lin(out, treatments, p)

        if self.alpha == 0.0:
            imbalance_loss = 0.0
        else:
            samples_loss = SamplesLoss(loss="sinkhorn", p=2, blur=self.blur, backend="tensorized")
            phi0 = out[mask0]
            phi1 = out[mask1]
            imbalance_loss = samples_loss(phi1, phi0)

        return (1.0 - p) * loss_t0 + p * loss_t1 + self.alpha * imbalance_loss


class BITES(BITES_Base):
    """Network structure similar to the DeepHit paper, but without the residual
    connections (for simplicity).
    """

    def __init__(self, in_features, num_nodes_shared, num_nodes_indiv, out_features,
                 num_treatments=2, batch_norm=True, dropout=None):
        super().__init__()
        self.shared_net = MLPVanilla(
            in_features, num_nodes_shared[:-1], num_nodes_shared[-1],
            batch_norm, dropout,
        )
        self.risk_nets = torch.nn.ModuleList()
        for _ in range(2):
            net = MLPVanilla(
                num_nodes_shared[-1], num_nodes_indiv, out_features,
                batch_norm, dropout,output_bias=False)
            self.risk_nets.append(net)

        self.baseline_hazards_ = []
        self.baseline_cumulative_hazards_ = []

    def forward(self, input, treatment):
        # treatment=input[:,-1]
        N = treatment.shape[0]
        y = torch.zeros_like(treatment)

        out = self.shared_net(input)

        out0 = out[treatment == 0]
        out1 = out[treatment == 1]

        out0 = self.risk_nets[0](out0)
        out1 = self.risk_nets[1](out1)

        k, j = 0, 0
        for i in range(N):
            if treatment[i] == 0:
                y[i] = out0[k]
                k = k + 1
            else:
                y[i] = out1[j]
                j = j + 1

        return y, out

    def predict(self, X, treatment):
        self.eval()
        out = self(X, treatment)
        self.train()
        return out[0], out[1]

    def predict_numpy(self, X, treatment):
        self.eval()
        X = torch.Tensor(X)
        treatment = torch.Tensor(treatment)
        out = self(X, treatment)
        self.train()
        return out[0].detach(), out[1].detach()











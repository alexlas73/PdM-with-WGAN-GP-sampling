"""The two WGAN-GP samplers compared in the study (imbalanced-learn compatible).

OneTabGANSampler          Single WGAN-GP: one generator trained on all training failures (FailureType ignored,
                          so failures without a recorded mechanism are included).
SpecializedTabGANSampler  Multi WGAN-GP (failure-type-aware): one independent generator per recorded mechanism;
                          the synthetic records needed for a 1:1 class ratio are allocated in proportion to each
                          mechanism's training failures, with largest-remainder rounding. Failures without a
                          recorded mechanism stay in the training data but train no generator.

Both expect X to contain the raw feature columns plus 'FailureType'; both return the original rows followed by
the synthetic rows, and keep the synthetic rows in .last_synth_df_ (used for the t-SNE figures).
The class bodies are unchanged from the code that produced the archived predictions.
"""
import numpy as np
import pandas as pd
from imblearn.base import BaseSampler

from . import compat

compat.apply()
try:
    from tabgan.sampler import GANGenerator
except ImportError:          # allows importing the module (e.g. for tests) without tabgan installed
    GANGenerator = None


class SpecializedTabGANSampler(BaseSampler):
    """
    Imbalanced-learn compatible sampler that:
      - splits minority class by FailureType
      - trains a separate TabGAN per FailureType subset
      - generates exactly enough synthetic minority rows to hit target_pos_ratio
    Expects X to include a 'FailureType' column and a 'Type' categorical column.
    """
    def __init__(self, gen_params, adv_params, categorical_features=None,
                 target_pos_ratio=0.5, failure_col='FailureType', random_state=42):
        self.gen_params = gen_params
        self.adv_params = adv_params
        self.categorical_features = categorical_features
        self.target_pos_ratio = target_pos_ratio
        self.failure_col = failure_col
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def _fit_resample(self, X, y):
        X_df = pd.DataFrame(X).reset_index(drop=True)
        y_sr = pd.Series(y).reset_index(drop=True)
        if self.failure_col not in X_df.columns:
            raise ValueError(f"'{self.failure_col}' must be present in X for specialized multi-GAN.")

        # Split by class
        Xy = X_df.copy(); Xy['__y__'] = y_sr
        minority_all = Xy[Xy['__y__'] == 1].drop(columns='__y__')
        majority_all = Xy[Xy['__y__'] == 0].drop(columns='__y__')
        n0, n1 = len(majority_all), len(minority_all)

        # How many synthetic 1s we need to reach the target ratio
        r = float(self.target_pos_ratio)
        if not 0 < r < 1:
            raise ValueError("target_pos_ratio must be in (0,1)")
        total_k = int(np.ceil((r/(1.0 - r))*n0 - n1))

        if total_k <= 0 or n1 == 0:
            # nothing to do
            self.last_synth_df_ = pd.DataFrame()
            self.last_X_aug_ = X_df.to_numpy()
            self.last_y_aug_ = y_sr.to_numpy()
            return X_df.to_numpy(), y_sr.to_numpy()

        # Which failure types to model
        bad = {'No Failure', 'Random Failures'}
        ft_counts = minority_all[self.failure_col].value_counts()
        ft_counts = ft_counts[[ft for ft in ft_counts.index if ft not in bad]]
        if ft_counts.empty:
            # fallback: single-GAN on all minority rows (rare)
            fts = []
        else:
            fts = list(ft_counts.index)

        synth_parts = []
        rng = np.random.RandomState(self.random_state)

        if not fts:
            # Single-GAN fallback
            subset = minority_all.copy()
            k_need = total_k
            run_gen_params = dict(self.gen_params)
            bs = min(run_gen_params.get('batch_size', 64), len(subset))
            if bs % 10 != 0: bs = max(10, (bs // 10) * 10)
            run_gen_params['batch_size'] = bs

            existing_cols = list(subset.columns)
            cat_cols = [c for c in (self.categorical_features or []) if c in existing_cols]

            sampler = GANGenerator(
                gen_x_times=max(k_need / float(len(subset)), 1e-6),
                cat_cols=cat_cols,
                is_post_process=True,
                adversarial_model_params=self.adv_params,
                gen_params=run_gen_params
            )
            y_min = pd.Series([1]*len(subset), name='Target', index=subset.index)
            train_df = subset.drop(columns=[self.failure_col], errors='ignore')
            synth_df, _ = sampler.generate_data_pipe(
                train_df=train_df, target=y_min.to_frame(),
                test_df=None, only_generated_data=True
            )
            # trim/pad to exact k
            if len(synth_df) > k_need:
                synth_df = synth_df.sample(n=k_need, random_state=self.random_state)
            elif len(synth_df) < k_need:
                extra = synth_df.sample(n=(k_need - len(synth_df)), replace=True,
                                        random_state=self.random_state)
                synth_df = pd.concat([synth_df, extra], ignore_index=True)
            # put back FailureType for bookkeeping
            synth_df[self.failure_col] = subset[self.failure_col].iloc[0]
            synth_parts.append(synth_df)

        else:
            # Proportional allocation by failure type with rounding fix
            prop = ft_counts / ft_counts.sum()
            raw = prop * total_k
            k_floor = np.floor(raw).astype(int)
            remainder = total_k - int(k_floor.sum())
            order = (raw - k_floor).sort_values(ascending=False)
            for ft in order.index[:remainder]:
                k_floor.loc[ft] += 1

            for ft, k_need in k_floor.items():
                if k_need <= 0:
                    continue
                subset = minority_all[minority_all[self.failure_col] == ft]
                run_gen_params = dict(self.gen_params)
                bs = min(run_gen_params.get('batch_size', 64), len(subset))
                if bs % 10 != 0: bs = max(10, (bs // 10) * 10)
                run_gen_params['batch_size'] = bs

                existing_cols = list(subset.columns)
                cat_cols = [c for c in (self.categorical_features or []) if c in existing_cols]

                sampler = GANGenerator(
                    gen_x_times=max(k_need / float(len(subset)), 1e-6),
                    cat_cols=cat_cols,
                    is_post_process=True,
                    adversarial_model_params=self.adv_params,
                    gen_params=run_gen_params
                )
                y_min = pd.Series([1]*len(subset), name='Target', index=subset.index)
                train_df = subset.drop(columns=[self.failure_col], errors='ignore')
                synth_df, _ = sampler.generate_data_pipe(
                    train_df=train_df, target=y_min.to_frame(),
                    test_df=None, only_generated_data=True
                )
                # trim/pad to exact k per type
                if len(synth_df) > k_need:
                    synth_df = synth_df.sample(n=k_need, random_state=self.random_state)
                elif len(synth_df) < k_need:
                    extra = synth_df.sample(n=(k_need - len(synth_df)), replace=True,
                                            random_state=self.random_state)
                    synth_df = pd.concat([synth_df, extra], ignore_index=True)
                # annotate failure type for traceability
                synth_df[self.failure_col] = ft
                synth_parts.append(synth_df)

        synth_all = pd.concat(synth_parts, ignore_index=True) if synth_parts else pd.DataFrame()
        # Align columns to original X
        synth_all = synth_all.reindex(columns=X_df.columns, fill_value=np.nan)

        X_aug = pd.concat([X_df, synth_all], ignore_index=True)
        y_aug = np.concatenate([y_sr.to_numpy(), np.ones(len(synth_all), dtype=y_sr.dtype)])

        # keep for visualization / debugging
        self.last_synth_df_ = synth_all.copy()
        self.last_X_aug_ = X_aug.to_numpy()
        self.last_y_aug_ = y_aug
        return X_aug, y_aug

    # old imblearn compatibility
    def fit_resample(self, X, y):
        return self._fit_resample(X, y)


class OneTabGANSampler(BaseSampler):
    def __init__(self, gen_params, adv_params, categorical_features=None,
                 target_pos_ratio=0.5, failure_col='FailureType', random_state=42):
        # IMPORTANT: do not modify/copy these; store exactly what was passed in.
        self.gen_params = gen_params
        self.adv_params = adv_params
        self.categorical_features = categorical_features
        self.target_pos_ratio = target_pos_ratio
        self.failure_col = failure_col
        self.random_state = random_state


    def fit(self, X, y):
        return self

    def _fit_resample(self, X, y):
        X_df = pd.DataFrame(X).reset_index(drop=True)
        y_sr = pd.Series(y).reset_index(drop=True)

        # counts
        Xy = X_df.copy(); Xy['__y__'] = y_sr
        minority = Xy[Xy['__y__'] == 1].drop(columns='__y__')
        majority = Xy[Xy['__y__'] == 0].drop(columns='__y__')
        n0, n1 = len(majority), len(minority)

        # how many synthetic 1s to reach the target ratio
        r = self.target_pos_ratio
        if not 0 < r < 1:
            raise ValueError("target_pos_ratio must be in (0,1)")
        k_needed = int(np.ceil((r / (1.0 - r)) * n0 - n1))

        if k_needed <= 0 or n1 == 0:
            # nothing to add
            self.last_synth_df_ = pd.DataFrame(columns=X_df.columns)
            self.last_X_aug_ = X_df.to_numpy()
            self.last_y_aug_ = y_sr.to_numpy()
            return X_df.to_numpy(), y_sr.to_numpy()

        # GAN works on minority features, ignoring FailureType if present
        minority_for_gan = minority.drop(columns=[self.failure_col], errors='ignore')
        run_gen_params = dict(self.gen_params)
        bs = min(run_gen_params.get('batch_size', 64), len(minority_for_gan))
        if bs % 10 != 0: bs = max(10, (bs // 10) * 10)
        run_gen_params['batch_size'] = bs

        cat_cols = [c for c in self.categorical_features if c in minority_for_gan.columns]

        sampler = GANGenerator(
            gen_x_times=max(k_needed / float(len(minority_for_gan)), 1e-6),
            cat_cols=cat_cols,
            is_post_process=True,
            adversarial_model_params=self.adv_params,
            gen_params=run_gen_params
        )
        y_min = pd.Series([1]*len(minority_for_gan), name='Target', index=minority_for_gan.index)
        synth_df, _ = sampler.generate_data_pipe(
            train_df=minority_for_gan,
            target=y_min.to_frame(),
            test_df=None,
            only_generated_data=True
        )

        # trim/pad to exactly k_needed
        if len(synth_df) > k_needed:
            synth_df = synth_df.sample(n=k_needed, random_state=self.random_state)
        elif len(synth_df) < k_needed:
            extra = synth_df.sample(n=(k_needed - len(synth_df)), replace=True, random_state=self.random_state)
            synth_df = pd.concat([synth_df, extra], ignore_index=True)

        # if FailureType column exists in the original X, add it as NaN for synthetic rows
        if self.failure_col in X_df.columns and self.failure_col not in synth_df.columns:
            synth_df[self.failure_col] = np.nan

        # align columns
        synth_df = synth_df.reindex(columns=X_df.columns, fill_value=np.nan)

        X_aug = pd.concat([X_df, synth_df], ignore_index=True)
        y_aug = np.concatenate([y_sr.to_numpy(), np.ones(len(synth_df), dtype=y_sr.dtype)])

        # keep for later visualizations
        self.last_synth_df_ = synth_df.copy()
        self.last_X_aug_ = X_aug.to_numpy()
        self.last_y_aug_ = y_aug
        return X_aug, y_aug

    # old imblearn compatibility
    def fit_resample(self, X, y):
        return self._fit_resample(X, y)

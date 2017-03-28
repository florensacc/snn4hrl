import collections

import numpy as np
import theano
import theano.tensor as TT

import rllab.misc.logger as logger
from rllab.algos.batch_polopt import BatchPolopt, BatchSampler
from rllab.algos.npo import NPO
from rllab.misc import ext
from rllab.misc.overrides import overrides
from rllab.sampler import parallel_sampler
from sandbox.snn4hrl.distributions.categorical import from_index, from_onehot
from sandbox.snn4hrl.regressors.latent_regressor import Latent_regressor
from sandbox.snn4hrl.sampler.utils import rollout
from sandbox.snn4hrl.sampler.utils_snn import rollout_snn


class BatchSampler_snn(BatchSampler):
    """
    Allows giving bonus for MI and other bonus_evaluators, hallucinate if needed (not used in the paper)
    and switching latent every certain number of time-steps.
    """

    def __init__(self,
                 *args,  # this collects algo, passing it to BatchSampler in the super __init__
                 bonus_evaluator=None,  # list of bonus evals
                 reward_coef_bonus=None,  # this is the total bonus from the bonus evaluator. it's a LIST
                 latent_regressor=None,  # Latent_regressor object for MI. Provides logging AND bonus if needed
                 reward_regressor_mi=0,  # this is for the regressor bonus, not the grid
                 self_normalize=False,  # this is for the hallucinated samples importance weight
                 switch_lat_every=0,
                 **kwargs
                 ):
        super(BatchSampler_snn, self).__init__(*args, **kwargs)  # this should be giving a self.algo
        self.bonus_evaluator = bonus_evaluator if bonus_evaluator else []
        self.reward_coef_bonus = reward_coef_bonus if reward_coef_bonus else [0] * len(self.bonus_evaluator)
        self.reward_regressor_mi = reward_regressor_mi
        self.latent_regressor = latent_regressor
        self.self_normalize = self_normalize
        self.switch_lat_every = switch_lat_every

    def _worker_collect_one_path_snn(self, G, max_path_length, switch_lat_every=0, scope=None):
        G = parallel_sampler._get_scoped_G(G, scope)
        path = rollout_snn(G.env, G.policy, max_path_length, switch_lat_every=switch_lat_every)
        return path, len(path["rewards"])

    def sample_paths(
            self,
            policy_params,
            max_samples,
            max_path_length=np.inf,
            env_params=None,
            scope=None):
        """
        :param policy_params: parameters for the policy. This will be updated on each worker process
        :param max_samples: desired maximum number of samples to be collected. The actual number of collected samples
        might be greater since all trajectories will be rolled out either until termination or until max_path_length is
        reached
        :param max_path_length: horizon / maximum length of a single trajectory
        :return: a list of collected paths
        """
        parallel_sampler.singleton_pool.run_each(
            parallel_sampler._worker_set_policy_params,
            [(policy_params, scope)] * parallel_sampler.singleton_pool.n_parallel
        )
        if env_params is not None:
            parallel_sampler.singleton_pool.run_each(
                parallel_sampler._worker_set_env_params,
                [(env_params, scope)] * parallel_sampler.singleton_pool.n_parallel
            )

        return parallel_sampler.singleton_pool.run_collect(
            # parallel_sampler._worker_collect_one_path_snn,  # now this is defined in parallel_sampler also!
            self._worker_collect_one_path_snn,  # now this is defined in parallel_sampler also!
            threshold=max_samples,
            args=(max_path_length, self.switch_lat_every, scope),
            show_prog_bar=True
        )

    def obtain_samples(self, itr):
        cur_params = self.algo.policy.get_param_values()
        paths = self.sample_paths(  # use the sample function above
            policy_params=cur_params,
            max_samples=self.algo.batch_size,
            max_path_length=self.algo.max_path_length,
            scope=self.algo.scope,
        )
        if self.algo.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(paths, self.algo.batch_size)
            return paths_truncated

    @overrides
    def process_samples(self, itr, paths):
        # count visitations or whatever the bonus wants to do. This should not modify the paths
        for b_eval in self.bonus_evaluator:
            logger.log("fitting bonus evaluator before processing...")
            b_eval.fit_before_process_samples(paths)
            logger.log("fitted")
        # save real undiscounted reward before changing them
        undiscounted_returns = [sum(path["rewards"]) for path in paths]
        logger.record_tabular('TrueAverageReturn', np.mean(undiscounted_returns))
        for path in paths:
            path['true_rewards'] = list(path['rewards'])

        # If using a latent regressor (and possibly adding MI to the reward):
        if isinstance(self.latent_regressor, Latent_regressor):
            with logger.prefix(' Latent_regressor '):
                self.latent_regressor.fit(paths)

                if self.reward_regressor_mi:
                    for i, path in enumerate(paths):
                        path['logli_latent_regressor'] = self.latent_regressor.predict_log_likelihood(
                            [path], [path['agent_infos']['latents']])[0]  # this is for paths usually..

                        path['rewards'] += self.reward_regressor_mi * path[
                            'logli_latent_regressor']  # the logli of the latent is the variable of the mutual information

        # for the extra bonus
        for b, b_eval in enumerate(self.bonus_evaluator):
            for i, path in enumerate(paths):
                bonuses = b_eval.predict(path)
                path['rewards'] += self.reward_coef_bonus[b] * bonuses

        real_samples = ext.extract_dict(
            BatchSampler.process_samples(self, itr, paths),
            # I don't need to process the hallucinated samples: the R, A,.. same!
            "observations", "actions", "advantages", "env_infos", "agent_infos"
        )
        real_samples["importance_weights"] = np.ones_like(real_samples["advantages"])

        return real_samples

    def log_diagnostics(self, paths):
        for b_eval in self.bonus_evaluator:
            b_eval.log_diagnostics(paths)

        if isinstance(self.latent_regressor, Latent_regressor):
            with logger.prefix(' Latent regressor logging | '):
                self.latent_regressor.log_diagnostics(paths)


class NPO_snn(NPO):
    """
    Natural Policy Optimization for SNNs:
    - differentiable reward bonus for L2 or KL between conditional distributions (commented out: not used in paper).
    - allows to give rewards for serveral divergence metrics among conditional distributions (through BatchSampler_snn)
    - logg individually for every latent as well as some "hierarchy" metric or the deterministic policy
    """

    def __init__(
            self,
            # some extra logging. What of this could be included in the sampler?
            log_individual_latents=False,  # to log the progress of each individual latent
            log_deterministic=False,  # log the performance of the policy with std=0 (for each latent separate)
            log_hierarchy=False,
            bonus_evaluator=None,
            reward_coef_bonus=None,
            latent_regressor=None,
            reward_regressor_mi=0,  # kwargs to the sampler (that also processes)
            switch_lat_every=0,
            **kwargs):
        # some logging
        self.log_individual_latents = log_individual_latents
        self.log_deterministic = log_deterministic
        self.log_hierarchy = log_hierarchy

        sampler_cls = BatchSampler_snn
        sampler_args = {'switch_lat_every': switch_lat_every,
                        'latent_regressor': latent_regressor,
                        'bonus_evaluator': bonus_evaluator,
                        'reward_coef_bonus': reward_coef_bonus,
                        'reward_regressor_mi': reward_regressor_mi,
                        }
        super(NPO_snn, self).__init__(sampler_cls=sampler_cls, sampler_args=sampler_args, **kwargs)

    @overrides
    def init_opt(self):
        assert not self.policy.recurrent
        is_recurrent = int(self.policy.recurrent)

        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )

        latent_var = self.policy.latent_space.new_tensor_variable(
            'latents',
            extra_dims=1 + is_recurrent,
        )

        advantage_var = ext.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=theano.config.floatX
        )
        dist = self.policy.distribution  # this can still be the dist P(a|s,__h__)
        old_dist_info_vars = {
            k: ext.new_tensor(
                'old_%s' % k,  # define tensors old_mean and old_log_std
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in dist.dist_info_keys
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]  ##put 2 tensors above in a list

        if is_recurrent:
            valid_var = TT.matrix('valid')
        else:
            valid_var = None

        dist_info_vars = self.policy.dist_info_sym(obs_var, latent_var)

        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        if is_recurrent:
            mean_kl = TT.sum(kl * valid_var) / TT.sum(valid_var)
            surr_loss = - TT.sum(lr * advantage_var * valid_var) / TT.sum(valid_var)
        else:
            mean_kl = TT.mean(kl)
            surr_loss = - TT.mean(lr * advantage_var)

        loss = surr_loss

        input_list = [  # these are sym var. the inputs in optimize_policy have to be in same order!
                         obs_var,
                         action_var,
                         advantage_var,
                         latent_var,
                     ] + old_dist_info_vars_list  # provide old mean and var, for the new states as they were sampled from it!
        if is_recurrent:
            input_list.append(valid_var)

        self.optimizer.update_opt(
            loss=loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )
        return dict()

    @overrides
    def optimize_policy(self, itr,
                        samples_data):  # make that samples_data comes with latents: see train in batch_polopt
        all_input_values = tuple(ext.extract(  # it will be in agent_infos!!! under key "latents"
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        all_input_values += (agent_infos[
                                 "latents"],)  # latents has already been processed and is the concat of all latents, but keeps key "latents"
        info_list = [agent_infos[k] for k in
                     self.policy.distribution.dist_info_keys]  # these are the mean and var used at rollout, corresponding to
        all_input_values += tuple(info_list)  # old_dist_info_vars_list as symbolic var
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)

        loss_before = self.optimizer.loss(all_input_values)
        # this should always be 0. If it's not there is a problem.
        mean_kl_before = self.optimizer.constraint_val(all_input_values)
        logger.record_tabular('MeanKL_Before', mean_kl_before)

        with logger.prefix(' PolicyOptimize | '):
            self.optimizer.optimize(all_input_values)

        mean_kl = self.optimizer.constraint_val(all_input_values)
        loss_after = self.optimizer.loss(all_input_values)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    @overrides
    def log_diagnostics(self, paths):
        BatchPolopt.log_diagnostics(self, paths)
        self.sampler.log_diagnostics(paths)

        if self.policy.latent_dim:

            if self.log_individual_latents and not self.policy.resample:  # this is only valid for finite discrete latents!!
                all_latent_avg_returns = []
                clustered_by_latents = collections.OrderedDict()  # this could be done within the distribution to be more general, but ugly
                for lat_key in range(self.policy.latent_dim):
                    clustered_by_latents[lat_key] = []
                for path in paths:
                    lat = path['agent_infos']['latents'][0]
                    lat_key = int(from_onehot(lat))  # from_onehot returns an axis less than the input.
                    clustered_by_latents[lat_key].append(path)

                for latent_key, paths in clustered_by_latents.items():  # what to do if this is empty?? set a default!
                    with logger.tabular_prefix(str(latent_key)), logger.prefix(str(latent_key)):
                        if paths:
                            undiscounted_rewards = [sum(path["true_rewards"]) for path in paths]
                        else:
                            undiscounted_rewards = [0]
                        all_latent_avg_returns.append(np.mean(undiscounted_rewards))
                        logger.record_tabular('Avg_TrueReturn', np.mean(undiscounted_rewards))
                        logger.record_tabular('Std_TrueReturn', np.std(undiscounted_rewards))
                        logger.record_tabular('Max_TrueReturn', np.max(undiscounted_rewards))
                        if self.log_deterministic:
                            lat = from_index(latent_key, self.policy.latent_dim)
                            with self.policy.fix_latent(lat), self.policy.set_std_to_0():
                                path_det = rollout(self.env, self.policy, self.max_path_length)
                                logger.record_tabular('Deterministic_TrueReturn', np.sum(path_det["rewards"]))

                with logger.tabular_prefix('all_lat_'), logger.prefix('all_lat_'):
                    logger.record_tabular('MaxAvgReturn', np.max(all_latent_avg_returns))
                    logger.record_tabular('MinAvgReturn', np.min(all_latent_avg_returns))
                    logger.record_tabular('StdAvgReturn', np.std(all_latent_avg_returns))

                if self.log_hierarchy:
                    max_in_path_length = 10
                    completed_in_paths = 0
                    path = rollout(self.env, self.policy, max_path_length=max_in_path_length, animated=False)
                    if len(path['rewards']) == max_in_path_length:
                        completed_in_paths += 1
                        for t in range(1, 50):
                            path = rollout(self.env, self.policy, max_path_length=10, animated=False,
                                           reset_start_rollout=False)
                            if len(path['rewards']) < 10:
                                break
                            completed_in_paths += 1
                    logger.record_tabular('Hierarchy', completed_in_paths)

        else:
            if self.log_deterministic:
                with self.policy.set_std_to_0():
                    path = rollout(self.env, self.policy, self.max_path_length)
                logger.record_tabular('Deterministic_TrueReturn', np.sum(path["rewards"]))

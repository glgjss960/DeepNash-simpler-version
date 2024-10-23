import enum 
import functools 
from typing import Any, Callable, Sequence, Tuple 
import chex 
import haiku as hk 
import jax 
from jax import lax 
from jax import numpy as jnp 
from jax import tree_util as tree 
import numpy as np 
import optax 
from open_spiel.python import policy as policy_lib 
import pyspiel 

Params = chex.ArrayTree 

class EntropySchedule: 
   def __init__(self, *, sizes: Sequence[int], repeats: Sequence[int]): 
      try:
        if len(repeats) != len(sizes): 
           raise ValueError("`repeats` must be parallel to `sizes`.") 
        if not sizes: 
           raise ValueError("`sizes` and `repeats` must not be empty.") 
        if any([(repeat <= 0) for repeat in repeats]): 
           raise ValueError("All repeat values must be strictly positive") 
        if repeats[-1] != 1: 
           raise ValueError("The last value in `repeats` must be equal to 1, " "ince the last iteration size is repeated forever.") 
      except ValueError as e: 
        raise ValueError( f"Entropy iteration schedule: repeats ({repeats}) and sizes ({sizes})." ) from e 
      schedule = [0] 
      for size, repeat in zip(sizes, repeats): 
        schedule.extend([schedule[-1] + (i + 1) * size for i in range(repeat)]) 
      self.schedule = np.array(schedule, dtype=np.int32) 
    
   def __call__(self, learner_step: int) -> Tuple[float, bool]: 
      last_size = self.schedule[-1] - self.schedule[-2] 
      last_start = self.schedule[-1] + ( learner_step - self.schedule[-1]) // last_size * last_size 
      start = jnp.amax(self.schedule * (self.schedule <= learner_step))
      finish = jnp.amin( self.schedule * (learner_step < self.schedule), initial=self.schedule[-1], where=(learner_step < self.schedule)) 
      size = finish - start 
      beyond = (self.schedule[-1] <= learner_step) 
      iteration_start = (last_start * beyond + start * (1 - beyond)) 
      iteration_size = (last_size * beyond + size * (1 - beyond)) 
      update_target_net = jnp.logical_and( learner_step > 0, jnp.sum(learner_step == iteration_start)) 
      alpha = jnp.minimum( (2.0 * (learner_step - iteration_start)) / iteration_size, 1.0) 
      return alpha, update_target_net 
    
@chex.dataclass(frozen=True) 
class FineTuning: 
        
        from_learner_steps: int = -1 
        policy_threshold: float = 0.03 
        policy_discretization: int = 32 
        
        def __call__(self, policy: chex.Array, mask: chex.Array, learner_steps: int) -> chex.Array: 
            chex.assert_equal_shape((policy, mask)) 
            do_finetune = jnp.logical_and(self.from_learner_steps >= 0, learner_steps > self.from_learner_steps) 
            return jnp.where(do_finetune, self.post_process_policy(policy, mask), policy) 
        
        def post_process_policy( self, policy: chex.Array, mask: chex.Array, ) -> chex.Array: 
            
            chex.assert_equal_shape((policy, mask)) 
            policy = self._threshold(policy, mask) 
            policy = self._discretize(policy) 
            return policy 
        
        def _threshold(self, policy: chex.Array, mask: chex.Array) -> chex.Array: 
           
           chex.assert_equal_shape((policy, mask))
           if self.policy_threshold <= 0: return policy 
           mask = mask * ( (policy >= self.policy_threshold) + (jnp.max(policy, axis=-1, keepdims=True) < self.policy_threshold)) 
           return mask * policy / jnp.sum(mask * policy, axis=-1, keepdims=True) 
        
        def _discretize(self, policy: chex.Array) -> chex.Array: 
           
           if self.policy_discretization <= 0: 
              return policy 
           
           if len(policy.shape) == 1: 
              return self._discretize_single(policy) 
           
           dims = len(policy.shape) - 1 
           vmapped = jax.vmap(self._discretize_single) 
           policy = hk.BatchApply(vmapped, num_dims=dims)(policy) 
           return policy 
        
        def _discretize_single(self, mu: chex.Array) -> chex.Array: 
            
            if len(mu.shape) == 2: 
                mu_ = jnp.squeeze(mu, axis=0) 
            else: 
                mu_ = mu 
            n_actions = mu_.shape[-1] 
            roundup = jnp.ceil(mu_ * self.policy_discretization).astype(jnp.int32) 
            result = jnp.zeros_like(mu_) 
            order = jnp.argsort(-mu_) 
            weight_left = self.policy_discretization 
            
            def f_disc(i, order, roundup, weight_left, result): 
                x = jnp.minimum(roundup[order[i]], weight_left) 
                result = jax.numpy.where(weight_left >= 0, result.at[order[i]].add(x), result) 
                weight_left -= x 
                return i + 1, order, roundup, weight_left, result 
            
            def f_scan_scan(carry, x): 
                i, order, roundup, weight_left, result = carry 
                i_next, order_next, roundup_next, weight_left_next, result_next = f_disc( i, order, roundup, weight_left, result) 
                carry_next = (i_next, order_next, roundup_next, weight_left_next, result_next)
                return carry_next, x 
            (_, _, _, weight_left_next, result_next), _ = jax.lax.scan( 
                f_scan_scan, 
                init=(jnp.asarray(0), order, roundup, weight_left, result), 
                xs=None, 
                length=n_actions) 
            result_next = jnp.where(weight_left_next > 0, 
                                    result_next.at[order[0]].add(weight_left_next), 
                                    result_next) 
            if len(mu.shape) == 2: 
                result_next = jnp.expand_dims(result_next, axis=0) 
            return result_next / self.policy_discretization 
        
        def _legal_policy(logits: chex.Array, legal_actions: chex.Array) -> chex.Array: 
            
            chex.assert_equal_shape((logits, legal_actions)) 
            
            l_min = logits.min(axis=-1, keepdims=True) 
            logits = jnp.where(legal_actions, logits, l_min) 
            logits -= logits.max(axis=-1, keepdims=True) 
            logits *= legal_actions exp_logits = jnp.where(legal_actions, jnp.exp(logits), 0) 
            exp_logits_sum = jnp.sum(exp_logits, axis=-1, keepdims=True) 
            return exp_logits / exp_logits_sum 
        
        def legal_log_policy(logits: chex.Array, legal_actions: chex.Array) -> chex.Array: 
            
            chex.assert_equal_shape((logits, legal_actions)) 

            logits_masked = logits + jnp.log(legal_actions) 
            max_legal_logit = logits_masked.max(axis=-1, keepdims=True) 
            logits_masked = logits_masked - max_legal_logit 
            
            exp_logits_masked = jnp.exp(logits_masked) 
            baseline = jnp.log(jnp.sum(exp_logits_masked, axis=-1, keepdims=True)) 
            
            log_policy = jnp.multiply(legal_actions, (logits - max_legal_logit - baseline)) 
            return log_policy 
        
        def _player_others(player_ids: chex.Array, valid: chex.Array, player: int) -> chex.Array: 
            
            chex.assert_equal_shape((player_ids, valid)) 
            current_player_tensor = (player_ids == player).astype(jnp.int32)
            res = 2 * current_player_tensor - 1 
            res = res * valid 
            return jnp.expand_dims(res, axis=-1) 
        
        def _policy_ratio(pi: chex.Array, mu: chex.Array, actions_oh: chex.Array, valid: chex.Array) -> chex.Array: 
            
            chex.assert_equal_shape((pi, mu, actions_oh)) 
            chex.assert_shape((valid,), actions_oh.shape[:-1]) 
            
            def _select_action_prob(pi): 
                return (jnp.sum(actions_oh * pi, axis=-1, keepdims=False) * valid + (1 - valid)) 
            
            pi_actions_prob = _select_action_prob(pi) 
            mu_actions_prob = _select_action_prob(mu) 
            return pi_actions_prob / mu_actions_prob 
        
        def _where(pred: chex.Array, true_data: chex.ArrayTree, false_data: chex.ArrayTree) -> chex.ArrayTree: 
            
            def _where_one(t, f): 
                chex.assert_equal_rank((t, f)) 
                
                p = jnp.reshape(pred, pred.shape + (1,) * (len(t.shape) - len(pred.shape))) 
                return jnp.where(p, t, f) 
            
            return tree.tree_map(_where_one, true_data, false_data) 
        
        def _has_played(valid: chex.Array, player_id: chex.Array, player: int) -> chex.Array: 
            
            chex.assert_equal_shape((valid, player_id)) 
            
            def _loop_has_played(carry, x): 
                valid, player_id = x 
                chex.assert_equal_shape((valid, player_id)) 
                our_res = jnp.ones_like(player_id) 
                opp_res = carry 
                reset_res = jnp.zeros_like(carry) 
                our_carry = carry 
                opp_carry = carry 
                reset_carry = jnp.zeros_like(player_id) 
                
                return _where(valid, _where((player_id == player), (our_carry, our_res), (opp_carry, opp_res)), (reset_carry, reset_res))
                
            _, result = lax.scan(f=_loop_has_played, init=jnp.zeros_like(player_id[-1]), xs=(valid, player_id), reverse=True) 
            return result 
        
        def v_trace( v: chex.Array, valid: chex.Array, player_id: chex.Array, acting_policy: chex.Array, merged_policy: chex.Array, merged_log_policy: chex.Array, player_others: chex.Array, actions_oh: chex.Array, reward: chex.Array, player: int, eta: float, lambda_: float, c: float, rho: float, ) -> Tuple[Any, Any, Any]: 
            
            gamma = 1.0 
            has_played = _has_played(valid, player_id, player) 
            policy_ratio = _policy_ratio(merged_policy, acting_policy, actions_oh, valid) 
            inv_mu = _policy_ratio( jnp.ones_like(merged_policy), acting_policy, actions_oh, valid) 
            eta_reg_entropy = (-eta * jnp.sum(merged_policy * merged_log_policy, axis=-1) * jnp.squeeze(player_others, axis=-1)) 
            eta_log_policy = -eta * merged_log_policy * player_others 
            @chex.dataclass(frozen=True) 
            class LoopVTraceCarry: 
                reward: chex.Array 
                
                reward_uncorrected: chex.Array 
                next_value: chex.Array 
                next_v_target: chex.Array 
                importance_sampling: chex.Array
            
            init_state_v_trace = LoopVTraceCarry( reward=jnp.zeros_like(reward[-1]), reward_uncorrected=jnp.zeros_like(reward[-1]), next_value=jnp.zeros_like(v[-1]), next_v_target=jnp.zeros_like(v[-1]), importance_sampling=jnp.ones_like(policy_ratio[-1])) 
            
            def _loop_v_trace(carry: LoopVTraceCarry, x) -> Tuple[LoopVTraceCarry, Any]: 
                (cs, player_id, v, reward, eta_reg_entropy, valid, inv_mu, actions_oh, eta_log_policy) = x 
                reward_uncorrected = ( reward + gamma * carry.reward_uncorrected + eta_reg_entropy) 
                discounted_reward = reward + gamma * carry.reward 
                our_v_target = ( v + jnp.expand_dims( jnp.minimum(rho, cs * carry.importance_sampling), axis=-1) * (jnp.expand_dims(reward_uncorrected, axis=-1) + gamma * carry.next_value - v) + lambda_ * jnp.expand_dims( jnp.minimum(c, cs * carry.importance_sampling), axis=-1) * gamma * (carry.next_v_target - carry.next_value)) 
                opp_v_target = jnp.zeros_like(our_v_target) 
                reset_v_target = jnp.zeros_like(our_v_target) 
                
                our_learning_output = ( v + eta_log_policy + actions_oh * jnp.expand_dims(inv_mu, axis=-1) * (jnp.expand_dims(discounted_reward, axis=-1) + gamma * jnp.expand_dims( carry.importance_sampling, axis=-1) * carry.next_v_target - v)) 
                opp_learning_output = jnp.zeros_like(our_learning_output) 
                reset_learning_output = jnp.zeros_like(our_learning_output) 
                
                our_carry = LoopVTraceCarry( reward=jnp.zeros_like(carry.reward), next_value=v, next_v_target=our_v_target, reward_uncorrected=jnp.zeros_like(carry.reward_uncorrected), importance_sampling=jnp.ones_like(carry.importance_sampling)) 
                opp_carry = LoopVTraceCarry( reward=eta_reg_entropy + cs * discounted_reward, reward_uncorrected=reward_uncorrected, next_value=gamma * carry.next_value, next_v_target=gamma * carry.next_v_target, importance_sampling=cs * carry.importance_sampling)
                reset_carry = init_state_v_trace 
                return _where(valid, _where((player_id == player), (our_carry, (our_v_target, our_learning_output)), (opp_carry, (opp_v_target, opp_learning_output))), (reset_carry, (reset_v_target, reset_learning_output))) 
            
            _, (v_target, learning_output) = lax.scan( f=_loop_v_trace, init=init_state_v_trace, xs=(policy_ratio, player_id, v, reward, eta_reg_entropy, valid, inv_mu, actions_oh, eta_log_policy), reverse=True) 
            return v_target, has_played, learning_output 
        
        def get_loss_v(v_list: Sequence[chex.Array], v_target_list: Sequence[chex.Array], mask_list: Sequence[chex.Array]) -> chex.Array: 
            
            chex.assert_trees_all_equal_shapes(v_list, v_target_list) 
            chex.assert_shape(mask_list, v_list[0].shape[:-1]) 
            loss_v_list = [] 
            for (v_n, v_target, mask) in zip(v_list, v_target_list, mask_list): 
              assert v_n.shape[0] == v_target.shape[0] 
              loss_v = jnp.expand_dims( mask, axis=-1) * (v_n - lax.stop_gradient(v_target))**2 
              normalization = jnp.sum(mask) 
              loss_v = jnp.sum(loss_v) / (normalization + (normalization == 0.0)) 
              loss_v_list.append(loss_v) 
            
            return sum(loss_v_list) 
        
        def apply_force_with_threshold(decision_outputs: chex.Array, force: chex.Array, threshold: float, threshold_center: chex.Array) -> chex.Array: 
            
            chex.assert_equal_shape((decision_outputs, force, threshold_center)) 
            can_decrease = decision_outputs - threshold_center > -threshold 
            can_increase = decision_outputs - threshold_center < threshold 
            force_negative = jnp.minimum(force, 0.0) 
            force_positive = jnp.maximum(force, 0.0) 
            clipped_force = can_decrease * force_negative + can_increase * force_positive 
            return decision_outputs * lax.stop_gradient(clipped_force) 
        
        def renormalize(loss: chex.Array, mask: chex.Array) -> chex.Array:
            chex.assert_equal_shape((loss, mask)) 
            loss = jnp.sum(loss * mask) 
            normalization = jnp.sum(mask) 
            return loss / (normalization + (normalization == 0.0)) 
        
        def get_loss_nerd(logit_list: Sequence[chex.Array], policy_list: Sequence[chex.Array], q_vr_list: Sequence[chex.Array], valid: chex.Array, player_ids: Sequence[chex.Array], legal_actions: chex.Array, importance_sampling_correction: Sequence[chex.Array], clip: float = 100, threshold: float = 2) -> chex.Array: 
            assert isinstance(importance_sampling_correction, list) 
            loss_pi_list = [] 
            for k, (logit_pi, pi, q_vr, is_c) in enumerate( zip(logit_list, policy_list, q_vr_list, importance_sampling_correction)): 
                assert logit_pi.shape[0] == q_vr.shape[0] 
                
                adv_pi = q_vr - jnp.sum(pi * q_vr, axis=-1, keepdims=True) 
                adv_pi = is_c * adv_pi 
                adv_pi = jnp.clip(adv_pi, a_min=-clip, a_max=clip) 
                adv_pi = lax.stop_gradient(adv_pi) 
                logits = logit_pi - jnp.mean( logit_pi * legal_actions, axis=-1, keepdims=True) 
                threshold_center = jnp.zeros_like(logits) 
                nerd_loss = jnp.sum( legal_actions * apply_force_with_threshold(logits, adv_pi, threshold, threshold_center), axis=-1) 
                nerd_loss = -renormalize(nerd_loss, valid * (player_ids == k)) 
                loss_pi_list.append(nerd_loss) 
            
            return sum(loss_pi_list) 
        
        @chex.dataclass(frozen=True) 
        class AdamConfig: 
            
            b1: float = 0.0 
            b2: float = 0.999 
            eps: float = 10e-8 
            
        @chex.dataclass(frozen=True) 
        class NerdConfig: 
            
            beta: float = 2.0
            clip: float = 10_000 
            
        class StateRepresentation(str, enum.Enum): 
            
            INFO_SET = "info_set" 
            OBSERVATION = "observation" 
            
        @chex.dataclass(frozen=True) 
        class RNaDConfig: 
            
            game_name: str 
            trajectory_max: int = 10 
            state_representation: StateRepresentation = StateRepresentation.INFO_SET 
            policy_network_layers: Sequence[int] = (256, 256) 
            batch_size: int = 256 
            learning_rate: float = 0.00005 
            adam: AdamConfig = AdamConfig() 
            clip_gradient: float = 10_000 
            target_network_avg: float = 0.001 
            entropy_schedule_repeats: Sequence[int] = (1,) 
            entropy_schedule_size: Sequence[int] = (20_000,) 
            eta_reward_transform: float = 0.2 
            nerd: NerdConfig = NerdConfig() 
            c_vtrace: float = 1.0 
            finetune: FineTuning = FineTuning() 
            seed: int = 42 
            
        @chex.dataclass(frozen=True) 
        class EnvStep: 
            
            valid: chex.Array = () 
            obs: chex.Array = () 
            legal: chex.Array = ()
            player_id: chex.Array = () 
            rewards: chex.Array = () 
            def __eq__(self, other): 
                comp = (self.obs == other.obs).all() 
                comp_ = (self.obs == self.obs).all() 
                if ( comp==comp_ ).any()==False: 
                    return False 
                return True 
        
        @chex.dataclass(frozen=True) 
        class ActorStep: 
            
            action_oh: chex.Array = () 
            policy: chex.Array = () 
            rewards: chex.Array = () 
            
        @chex.dataclass(frozen=True) 
        class TimeStep: 
            
            env: EnvStep = EnvStep() 
            actor: ActorStep = ActorStep() 
            Optimizer = Callable[[Params, Params], Params] # (params, grads) -> params 
            def optax_optimizer( params: chex.ArrayTree, init_and_update: optax.GradientTransformation) -> Optimizer: 
                
                init_fn, update_fn = init_and_update 
                @chex.dataclass 
                class OptaxOptimizer: 
                    state: chex.Array 
                    def __call__(self, params: Params, grads: Params) -> Params: 
                        updates, self.state = update_fn(grads, self.state) 
                        return optax.apply_updates(params, updates) 
                    
                return OptaxOptimizer(state=init_fn(params)) 
            
        class RNaDSolver(policy_lib.Policy): 
            
            def __init__(self, config: RNaDConfig): 
                self.config = config 
                self.learner_steps = 0 s
                elf.actor_steps = 0
                self.init() 
                
            def init(self): 
                self._rngkey = jax.random.PRNGKey(self.config.seed)
                self._np_rng = np.random.RandomState(self.config.seed)                 
                self._game = pyspiel.load_game(self.config.game_name) 
                self._ex_state = self._play_chance(self._game.new_initial_state()) 
                
                def network( env_step: EnvStep ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]: 
                    mlp_torso = hk.nets.MLP(self.config.policy_network_layers) 
                    torso = mlp_torso(env_step.obs) 
                    mlp_policy_head = hk.nets.MLP([self._game.num_distinct_actions()]) 
                    logit = mlp_policy_head(torso) 
                    mlp_policy_value = hk.nets.MLP([1]) 
                    v = mlp_policy_value(torso) 
                    pi = _legal_policy(logit, env_step.legal) 
                    log_pi = legal_log_policy(logit, env_step.legal) 
                    return pi, v, log_pi, logit 
                
                self.network = hk.without_apply_rng(hk.transform(network)) 
                self._entropy_schedule = EntropySchedule( sizes=self.config.entropy_schedule_size, repeats=self.config.entropy_schedule_repeats) 
                self._loss_and_grad = jax.value_and_grad(self.loss, has_aux=False) 
                env_step = self._state_as_env_step(self._ex_state) 
                key = self._next_rng_key() self.params = self.network.init(key, env_step) 
                self.params_target = self.network.init(key, env_step) 
                self.params_prev = self.network.init(key, env_step) 
                self.params_prev_ = self.network.init(key, env_step) 
                self.optimizer = optax_optimizer( self.params, optax.chain( optax.scale_by_adam( eps_root=0.0, **self.config.adam, ), optax.scale(-self.config.learning_rate), optax.clip(self.config.clip_gradient))) 
                self.optimizer_target = optax_optimizer(self.params_target, optax.sgd(self.config.target_network_avg)) 
                
                self.vision_value=[] 
                self.vision_policy=[] 
                self.vision_policy_1=[] 
                self.vision_policy_2=[] 
                self.vision_log_policy=[] 
                self.vision_log_policy_1=[] 
                self.vision_log_policy_2=[] 
                self.vision_logit=[] 
                self.vision_logit_1=[] 
                self.vision_logit_2=[] 
                self.vision_params=[] 
                self.vision_params_target=[] 
                
            def vision_notes(self, params: Params, params_target: Params, ts: TimeStep, learner_steps: int): 
                    rollout = jax.vmap(self.network.apply, (None, 0), 0) 
                    pi, v, log_pi, logit = rollout(params, ts.env) 
                    self.vision_params.append(params) 
                    self.vision_params_target.append(params_target) 
                    return params, params_target, ts, learner_steps 
                
            def test_params(self,params:Sequence[Params])->Sequence[EnvStep]: 
                    
                    states = [ self._play_chance(self._game.new_initial_state()) for _ in range(100) ]
                    env_step_rec = [self._state_as_env_step(state) for state in states] 
                    for i in range(10): 
                        for _ in range(len(states)): 
                            pi, v, log_pi, logit = self.network.apply( self.network.init(self._next_rng_key(), env_step_rec[_]), env_step_rec[_]) 
                            pi = pi / np.sum(pi, axis=-1, keepdims=True) 
                            action = np.apply_along_axis( lambda x: self._np_rng.choice(range(pi.shape[0]), p=x), axis=-1, arr=pi) 
                            if not states[_].is_terminal(): 
                                states[_].apply_action(action) 
                                self._play_chance(states[_]) 
                                env_step_rec.append(self._state_as_env_step(states[_])) #states.append(states[_])
                    
                    res = env_step_rec  
                    env_step_rec = [] 
                    for _ in res: 
                      if _ not in env_step_rec: 
                        env_step_rec.append(_) 
                    
                    for env_step in env_step_rec: 
                      aux_pi = [] 
                      aux_pi_x = [] 
                      aux_pi_y = [] 
                      aux_v = [] 
                      aux_log_pi = [] 
                      aux_log_pi_x = [] 
                      aux_log_pi_y = [] 
                      aux_logit = [] 
                      aux_logit_x = [] 
                      aux_logit_y = [] 
                      for _ in range(len(params)): 
                        pi, v, log_pi, logit = self.network.apply(params[_], env_step) 
                        policy_pprocessed = self.config.finetune(pi, env_step.legal, _) 
                        aux_pi.append(policy_pprocessed) 
                        aux_pi_x.append(policy_pprocessed[0]) 
                        aux_pi_y.append(policy_pprocessed[1]) 
                        aux_v.append(v) 
                        aux_log_pi.append(log_pi) 
                        aux_log_pi_x.append(log_pi[0]) 
                        aux_log_pi_y.append(log_pi[1]) 
                        aux_logit.append(logit) 
                        aux_logit_x.append(logit[0]) 
                        aux_logit_y.append(logit[1]) 
                        #player_2.append(policy_pprocessed[1]) 
                      self.vision_policy.append(aux_pi) 
                      self.vision_policy_1.append(aux_pi_x) 
                      self.vision_policy_2.append(aux_pi_y) 
                      self.vision_value.append(aux_v) 
                      self.vision_log_policy.append(aux_log_pi) 
                      self.vision_log_policy_1.append(aux_log_pi_x) 
                      self.vision_log_policy_2.append(aux_log_pi_y) 
                      self.vision_logit.append(aux_logit) 
                      self.vision_logit_1.append(aux_logit_x) 
                      self.vision_logit_2.append(aux_logit_y) 
                      #self.vision_policy_2.append(player_2)
                    
                    return env_step_rec 
                
            def loss(self, params: Params, params_target: Params, params_prev: Params, params_prev_: Params, ts: TimeStep, alpha: float, learner_steps: int) -> float: 
                    
                    rollout = jax.vmap(self.network.apply, (None, 0), 0) 
                    pi, v, log_pi, logit = rollout(params, ts.env) 
                    policy_pprocessed = self.config.finetune(pi, ts.env.legal, learner_steps) 
                    _, v_target, _, _ = rollout(params_target, ts.env) 
                    _, _, log_pi_prev, _ = rollout(params_prev, ts.env) 
                    _, _, log_pi_prev_, _ = rollout(params_prev_, ts.env) 
                    
                    log_policy_reg = log_pi - (alpha * log_pi_prev + (1 - alpha) * log_pi_prev_) 
                    v_target_list, has_played_list, v_trace_policy_target_list = [], [], [] 
                    for player in range(self._game.num_players()): 
                        reward = ts.actor.rewards[:, :, player] 
                        v_target_, has_played, policy_target_ = v_trace( v_target, ts.env.valid, ts.env.player_id, ts.actor.policy, policy_pprocessed, log_policy_reg, _player_others(ts.env.player_id, ts.env.valid, player), ts.actor.action_oh, reward, player, lambda_=1.0, c=self.config.c_vtrace, rho=np.inf, eta=self.config.eta_reward_transform) 
                        v_target_list.append(v_target_) 
                        has_played_list.append(has_played) 
                        v_trace_policy_target_list.append(policy_target_) 
                    
                    loss_v = get_loss_v([v] * self._game.num_players(), v_target_list, has_played_list) 
                    is_vector = jnp.expand_dims(jnp.ones_like(ts.env.valid), axis=-1) 
                    importance_sampling_correction = [is_vector] * self._game.num_players() 
                    loss_nerd = get_loss_nerd( [logit] * self._game.num_players(), [pi] * self._game.num_players(), v_trace_policy_target_list, ts.env.valid,ts.env.player_id, ts.env.legal, importance_sampling_correction, clip=self.config.nerd.clip, threshold=self.config.nerd.beta) 
                    return loss_v + loss_nerd 
                
            @functools.partial(jax.jit, static_argnums=(0,)) 
            def update_parameters( self, params: Params, params_target: Params, params_prev: Params, params_prev_: Params, optimizer: Optimizer, optimizer_target: Optimizer, timestep: TimeStep, alpha: float, learner_steps: int, update_target_net: bool): 
                    
                    loss_val, grad = self._loss_and_grad(params, params_target, params_prev, params_prev_, timestep, alpha, learner_steps) 
                    params = optimizer(params, grad) 
                    params_target = optimizer_target( params_target, tree.tree_map(lambda a, b: a - b, params_target, params)) 
                    params_prev, params_prev_ = jax.lax.cond( update_target_net, lambda: (params_target, params_prev), lambda: (params_prev, params_prev_)) 
                    
                    logs = { "loss": loss_val, }
                    return (params, params_target, params_prev, params_prev_, optimizer, optimizer_target), logs 
                
            def __getstate__(self): 
                    return dict(config=self.config, learner_steps=self.learner_steps, actor_steps=self.actor_steps, np_rng=self._np_rng.get_state(), rngkey=self._rngkey, params=self.params, params_target=self.params_target, params_prev=self.params_prev, params_prev_=self.params_prev_, optimizer=self.optimizer.state, optimizer_target=self.optimizer_target.state, ) 
                
            def __setstate__(self, state): 
                    self.config = state["config"] 
                    self.init() 
                    self.learner_steps = state["learner_steps"] 
                    self.actor_steps = state["actor_steps"] 
                    self._np_rng.set_state(state["np_rng"]) 
                    self._rngkey = state["rngkey"] 
                    self.params = state["params"] 
                    self.params_target = state["params_target"] 
                    self.params_prev = state["params_prev"] 
                    self.params_prev_ = state["params_prev_"] 
                    self.optimizer.state = state["optimizer"] 
                    self.optimizer_target.state = state["optimizer_target"] 
                    
            def step(self): 
                    timestep = self.collect_batch_trajectory() 
                    vision_notes=self.vision_notes(self.params,self.params_target,timestep, self.learner_steps) 
                    alpha, update_target_net = self._entropy_schedule(self.learner_steps) 
                    (self.params, self.params_target, self.params_prev, self.params_prev_, self.optimizer, self.optimizer_target), logs = self.update_parameters( self.params, self.params_target, self.params_prev, self.params_prev_, self.optimizer, self.optimizer_target, timestep, alpha,self.learner_steps, update_target_net) 
                    self.learner_steps += 1 
                    logs.update({ "actor_steps": self.actor_steps, "learner_steps": self.learner_steps, })
                    return logs 
                
            def _next_rng_key(self) -> chex.PRNGKey: 
                    self._rngkey, subkey = jax.random.split(self._rngkey) 
                    return subkey 
                
            def _state_as_env_step(self, state: pyspiel.State) -> EnvStep: 
                    rewards = np.array(state.returns(), dtype=np.float64) 
                    valid = not state.is_terminal() 
                    if not valid: 
                        state = self._ex_state 
                    if self.config.state_representation == StateRepresentation.OBSERVATION: 
                        obs = state.observation_tensor() 
                    elif self.config.state_representation == StateRepresentation.INFO_SET: 
                        obs = state.information_state_tensor() 
                    else: raise ValueError( f"Invalid StateRepresentation: {self.config.state_representation}.") 
                    return EnvStep( obs=np.array(obs, dtype=np.float64), legal=np.array(state.legal_actions_mask(), dtype=np.int8), player_id=np.array(state.current_player(), dtype=np.float64), valid=np.array(valid, dtype=np.float64), rewards=rewards) 
                
            def action_probabilities(self, state: pyspiel.State, player_id: Any = None): 
                    env_step = self._batch_of_states_as_env_step([state]) 
                    probs = self._network_jit_apply_and_post_process( self.params_target, env_step) 
                    probs = jax.device_get(probs[0]) 
                    return { action: probs[action] for action, valid in enumerate(jax.device_get(env_step.legal[0])) if valid }
                
            @functools.partial(jax.jit, static_argnums=(0,)) 
            def _network_jit_apply_and_post_process( self, params: Params, env_step: EnvStep) -> chex.Array: 
                    pi, _, _, _ = self.network.apply(params, env_step) 
                    pi = self.config.finetune.post_process_policy(pi, env_step.legal) 
                    return pi 
                
            def actor_step(self, env_step: EnvStep): 
                    pi, _, _, _ = self.network.apply(self.params, env_step) 
                    pi = np.asarray(pi).astype("float64") 
                    pi = pi / np.sum(pi, axis=-1, keepdims=True) 
                    action = np.apply_along_axis( lambda x: self._np_rng.choice(range(pi.shape[1]), p=x), axis=-1, arr=pi) 
                    action_oh = np.zeros(pi.shape, dtype="float64") 
                    action_oh[range(pi.shape[0]), action] = 1.0 
                    actor_step = ActorStep(policy=pi, action_oh=action_oh, rewards=()) 
                    return action, actor_step 
                
            def collect_batch_trajectory(self) -> TimeStep: 
                    states = [ self._play_chance(self._game.new_initial_state()) for _ in range(self.config.batch_size) ]
                    timesteps = [] 
                    env_step = self._batch_of_states_as_env_step(states) 
                    for _ in range(self.config.trajectory_max): 
                        prev_env_step = env_step 
                        a, actor_step = self.actor_step(env_step) 
                        states = self._batch_of_states_apply_action(states, a) 
                        env_step = self._batch_of_states_as_env_step(states) 
                        timesteps.append( TimeStep( env=prev_env_step, actor=ActorStep( action_oh=actor_step.action_oh, policy=actor_step.policy, rewards=env_step.rewards), )) 
                        
                    return jax.tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *timesteps) 
                    
                    
            def _batch_of_states_as_env_step(self, states: Sequence[pyspiel.State]) -> EnvStep: 
                    envs = [self._state_as_env_step(state) for state in states]
                    return jax.tree_util.tree_map(lambda *e: np.stack(e, axis=0), *envs) 
            
            def _batch_of_states_apply_action( self, states: Sequence[pyspiel.State], actions: chex.Array) -> Sequence[pyspiel.State]: 
                for state, action in zip(states, list(actions)): 
                    if not state.is_terminal(): 
                        self.actor_steps += 1 
                        state.apply_action(action) 
                        self._play_chance(state) 
                
                return states 
            
            def _play_chance(self, state: pyspiel.State) -> pyspiel.State: 
                while state.is_chance_node(): 
                    chance_outcome, chance_proba = zip(*state.chance_outcomes()) 
                    action = self._np_rng.choice(chance_outcome, p=chance_proba) 
                    state.apply_action(action) 
                
                return state
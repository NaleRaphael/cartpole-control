from . import agent
from . import env
from . import model
from . import solver

__all__ = []
__all__.extend(agent.__all__)
__all__.extend(env.__all__)
__all__.extend(model.__all__)
__all__.extend(solver.__all__)


# Dirty patch to avoid annoying warning while creating `gym.spaces`
# without explicitly assigned dtype.
# goal: (@CartPoleEnv.__init__)
#     self.observation_space = spaces.Box(-high, high)
# ->  self.observation_space = spaces.Box(-high, high, dtype='float')
import gym
import sys

version2int =  lambda v: list(map(int, v.split('.')))
i2b = lambda x: (x).to_bytes(1, byteorder=sys.byteorder)

gv = version2int(gym.__version__)

if 1 >= gv[0] > 0 or gv[1] >= 9:
    import dis
    from types import CodeType
    from gym.envs.classic_control.cartpole import CartPoleEnv

    def dirty_patch(f):
        fc = f.__code__

        # additional kwargs for calling spaces.Box()
        patch_consts = ['dtype', 'float']

        consts = tuple(list(fc.co_consts) + patch_consts)
        stacksize = fc.co_stacksize + len(consts)
        payload = fc.co_code

        # ----- find position of function call -----
        idx_name_spaces = fc.co_names.index('spaces')
        idx_name_box = fc.co_names.index('Box')
        ins_end = i2b(0)

        # LOAD_GLOBAL        ? (spaces)
        # LOAD_ATTR          ? (Box)
        ins = (
            i2b(dis.opmap['LOAD_GLOBAL']) + i2b(idx_name_spaces) + ins_end +
            i2b(dis.opmap['LOAD_ATTR']) + i2b(idx_name_box) + ins_end
        )
        idx_call = payload.index(ins)

        # CALL_FUNCTION      ? (2 positional, 0 keyword pair)
        ins = (
            i2b(dis.opmap['CALL_FUNCTION']) + i2b(2) + i2b(0)
        )
        idx_call += payload[idx_call:].index(ins)
        # ------------------------------------------

        # edit payload
        payload = (
            payload[:idx_call] +
            # LOAD_CONST     ? (index of `dtype` in co_consts)
            i2b(dis.opmap['LOAD_CONST']) + i2b(len(consts)-2) + ins_end +
            # LOAD_CONST     ? (index of `float` in co_consts)
            i2b(dis.opmap['LOAD_CONST']) + i2b(len(consts)-1) + ins_end +
            # CALL_FUNCTION  ? (2 positional, 1 keyword pair)
            i2b(dis.opmap['CALL_FUNCTION']) + i2b(2) + i2b(1) +
            # skip original function call (3 bytes)
            payload[idx_call+3:]
        )

        f.__code__ = CodeType(
            fc.co_argcount,
            fc.co_kwonlyargcount,
            fc.co_nlocals,
            stacksize,
            fc.co_flags,
            payload,
            consts,
            fc.co_names,
            fc.co_varnames,
            fc.co_filename,
            fc.co_name,
            fc.co_firstlineno,
            fc.co_lnotab,
            fc.co_freevars,
            fc.co_cellvars,
        )
        return f

    try:
        CartPoleEnv.__init__ = dirty_patch(CartPoleEnv.__init__)
    except:
        sys.stderr.write('Failed to patch `CartPoleEnv.__init__`')
        raise
    finally:
        del dis, CodeType, CartPoleEnv, dirty_patch

del version2int, i2b, gv

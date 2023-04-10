from typing import NamedTuple, Sequence, Tuple, Union

# Global map that will contain registered mmwavecli cmds
MMWAVECLI_CMDS = {}

# Define explicit types
RegisteredCmd = NamedTuple
UnregisteredCmd = Tuple[str, tuple]
ParsedCmd = Union[RegisteredCmd, UnregisteredCmd]
ConfigCmds = Union[str, Sequence[ParsedCmd]]


def mmwavecli_cmd(cmd: str, is_cfg: bool = False):
    """
    Decorator for registering CLI commands dataclasses
    @param cmd: the command name
    @param is_cfg: whether it's a cfg command
    @return: the input type, after registration
    """
    def make_mmwavecli_cmd(cls: type):
        cls.__cmd__ = cmd
        cls.__is_cfg__ = is_cfg
        MMWAVECLI_CMDS[cmd] = cls
        return cls
    return make_mmwavecli_cmd


def mmwavecli_cfg(cmd: str):
    """
    Decorator for registering CLI config commands dataclasses
    @param cmd: the command name
    :return: the input type, after registration
    """
    return mmwavecli_cmd(cmd, is_cfg=True)


def parse_cmd(cmd: Union[str, ParsedCmd], *args) -> ParsedCmd:
    """
    Parse command into a registered type if available
    @param cmd: encoded command line, command name or already parsed cmd
    @param args: when "cmd" is just the command name this will be the sequence of args
    @return: parsed command, as object if registered or tuple otherwise
    """

    if isinstance(cmd, str):
        if len(args) == 0:  # str -> cmd, args
            args = cmd.split(' ')
            cmd = args.pop(0)
        else:
            pass  # cmd, args -> cmd, args
    elif hasattr(cmd, '__cmd__'):  # RegisteredCmd -> Cmd
        return cmd
    else:  # UnregisteredCmd -> cmd, args
        cmd, args = cmd

    if cmd not in MMWAVECLI_CMDS:
        # command not registered, (cmd, args) -> UnregisteredCmd
        return cmd, tuple(args)

    cmd_cls = MMWAVECLI_CMDS[cmd]

    args_types = cmd_cls.__annotations__.values()
    casted_args = (t(a) for t, a in zip(args_types, args))
    return cmd_cls(*casted_args)  # (cmd, args) -> RegisteredCmd


def encode_cmd(cmd: Union[ParsedCmd, str], *args) -> str:
    """
    Encodes a command object into a command line
    @param cmd: command to be encoded as command line
    @return: the configuration line
    """

    if isinstance(cmd, str):
        pass  # cmd, args -> cmd, args
    elif hasattr(cmd, "__cmd__"):  # RegisteredCmd -> cmd, args
        args = tuple(str(arg) for arg in cmd)
        cmd = cmd.__cmd__
    elif isinstance(cmd, Sequence):  # UnregisteredCmd|tuple -> cmd, args
        args = isinstance(cmd[1], Sequence) and cmd[1] or cmd[1:]
        cmd = cmd[0]

    args = (cmd,) + tuple(str(arg) for arg in args)
    return " ".join(args)


def parse_config(config: ConfigCmds, cfg_only=True) -> Sequence[ParsedCmd]:
    """
    Parse config commands from file or sequence, into a list of config commands
    :param config: path to config file, or sequence of commands
    :param cfg_only: whether to filter out non-config commands (recommended)
    :return:
    """
    if isinstance(config, str):
        with open(config, "r") as fh:
            lines = [ln.strip() for ln in fh.readlines()]

            # Remove comments
            lines = [ln for ln in lines if not ln.startswith('%')]
            cmds = [parse_cmd(ln) for ln in lines]
    else:
        cmds = config

    # TODO: allow unparsed cmds sequence as input

    if cfg_only:
        # Note: considering unregistered commands as cfg for compatibility
        cmds = [c for c in cmds if getattr(c, '__is_cfg__', True)]

    return cmds

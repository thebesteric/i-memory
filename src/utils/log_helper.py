import logging
import logging.config
import os
import re
from importlib.resources import files
from typing import Any

import pyrootutils
import yaml


# === Required packages ===
# pip install pyyaml
# pip install colorlog
# =========================

class LogHelper:
    """
    日志工具类
    """
    # 默认日志实例（用于内部日志记录）
    _logger = logging.getLogger()

    # 日志实例（单例）
    _instances: dict[str, logging.Logger] = {}

    # 匹配{变量名}的正则（支持字母、数字、下划线）
    _var_pattern = re.compile(r"\{(\w+)\}")

    # 忽略的配置键（logging 模块原生配置项）
    NATIVE_KEYS = {"version", "disable_existing_loggers", "formatters", "handlers", "loggers", "root"}

    # 配置文件名称
    CONFIG_FILE_NAME = "agile_logger.yaml"

    @classmethod
    def load_config(cls):
        """
        读取包内置的配置文件
        :return:
        """
        # 读取并解析 YAML 文件
        project_root = pyrootutils.find_root()
        config_yaml_path = project_root / cls.CONFIG_FILE_NAME
        # 判断文件是否存在，不存在则读取默认配置
        if not os.path.exists(config_yaml_path):
            # 获取 YAML 文件的绝对路径
            cls._logger.warning("使用默认日志配置文件")
            config_yaml_path = files("agile_commons.config").joinpath(cls.CONFIG_FILE_NAME)
        # 读取并解析 YAML 文件
        with open(str(config_yaml_path), "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config

    @classmethod
    def get_logger(cls, name: str | None = None) -> logging.Logger:
        """
        获取日志实例（单例）
        @param name: 日志实例名称
        @param config_path: 日志配置文件路径（相对于项目根目录的路径）
        @return: 日志实例
        """
        if name in cls._instances:
            return cls._instances[name]

        if name is None:
            import inspect
            caller_frame = inspect.stack()[1]
            caller_module = inspect.getmodule(caller_frame[0])
            name = caller_module.__name__ if caller_module else "unknown"

        try:
            # 解析配置文件
            config = cls.load_config()

            # 提取 YAML 顶层所有自定义变量（排除 logging 模块原生配置键）
            variables = {k: v for k, v in config.items() if k not in cls.NATIVE_KEYS}

            # 处理 log_dir：拼接项目根目录，确保绝对路径
            project_root = pyrootutils.find_root()
            if "log_dir" in variables:
                log_dir = variables["log_dir"]
                # 相对路径 → 项目根目录/相对路径（绝对路径）
                log_dir_abs = project_root / log_dir
                # 转为字符串路径
                variables["log_dir"] = str(log_dir_abs)

            # 递归替换配置中所有{变量名}占位符
            cls._replace_variables(config, variables)

            # 确保日志目录存在
            if "log_dir" in variables:
                os.makedirs(variables["log_dir"], exist_ok=True)

            # 应用配置
            logging.config.dictConfig(config)
            cls._instances[name] = logging.getLogger(name)

        except Exception as e:
            cls._logger.warning(f"加载日志配置失败: {e}，使用默认配置")
            # 降级为默认控制台日志
            cls._instances[name] = logging.getLogger(name)
            cls._instances[name].setLevel(logging.DEBUG)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s")
            console_handler.setFormatter(formatter)
            cls._instances[name].addHandler(console_handler)

        return cls._instances[name]

    @classmethod
    def _replace_variables(cls, config: dict[str, Any], variables: dict[str, Any]):
        """
        递归遍历配置字典，替换所有 {变量名} 占位符
        @param config: 日志配置字典
        @param variables: 自定义变量字典
        """
        for key, value in config.items():
            if isinstance(value, str):
                # 替换字符串中的所有 {变量名}
                replaced_value = cls._var_pattern.sub(
                    # 找不到变量则保留原占位符
                    lambda m: str(variables.get(m.group(1), m.group(0))),
                    value
                )
                # 如果替换后是数字字符串，转换为整数/浮点数
                if replaced_value.isdigit():
                    config[key] = int(replaced_value)
                elif replaced_value.replace('.', '', 1).isdigit():
                    config[key] = float(replaced_value)
                else:
                    config[key] = replaced_value
            elif isinstance(value, dict):
                # 递归处理嵌套字典
                cls._replace_variables(value, variables)
            # 忽略列表和其他类型（logging 配置中列表通常是 handler 名称等，无需替换）

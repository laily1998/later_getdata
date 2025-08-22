# -*- coding: utf-8 -*-
# Author: Puyuan Du
import warnings
import datetime
from pathlib import Path
from collections import namedtuple, defaultdict
from typing import Union, Optional, List, Any, Generator, Dict
import pandas as pd
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import time
from cinrad.projection import get_coordinate, height
from cinrad.error import RadarDecodeError
from cinrad.io.base import RadarBase, prepare_file
from cinrad.io._dtype import *
from cinrad._typing import Number_T
import tqdm
__all__ = [ "StandardData"]

ScanConfig = namedtuple("ScanConfig", SDD_cut.fields.keys())
ScanConfigPA = namedtuple("ScanConfigPA", PA_SDD_cut.fields.keys())
utc_offset = datetime.timedelta(hours=8)

utc8_tz = datetime.timezone(utc_offset)


def epoch_seconds_to_utc(epoch_seconds: float) -> datetime.datetime:
    r"""Convert epoch seconds to UTC datetime"""
    return datetime.datetime.utcfromtimestamp(epoch_seconds).replace(
        tzinfo=datetime.timezone.utc
    )


def localdatetime_to_utc(
    ldt: datetime.datetime, tz: datetime.timezone = utc8_tz
) -> datetime.datetime:
    r"""Convert local datetime to UTC datetime"""
    if ldt.tzinfo is None:
        ldt = ldt.replace(tzinfo=tz)  # suppose the default timezone is UTC+8
    return ldt.astimezone(datetime.timezone.utc)


def vcp(el_num: int) -> str:
    r"""Determine volume coverage pattern by number of scans."""
    if el_num == 5:
        task_name = "VCP31"
    elif el_num == 9:
        task_name = "VCP21"
    elif el_num == 14:
        task_name = "VCP11"
    else:
        task_name = "Unknown"
    return task_name


def infer_type(f: Any, filename: str) -> tuple:
    r"""Detect radar type from records in file"""
    # Attempt to find information in file, which has higher
    # priority compared with information obtained from file name
    radartype = None
    code = None
    f.seek(100)
    typestring = f.read(9)
    if typestring == b"CINRAD/SC":
        radartype = "SC"
    elif typestring == b"CINRAD/CD":
        radartype = "CD"
    f.seek(116)
    if f.read(9) == b"CINRAD/CC":
        radartype = "CC"
    # Read information from filename (if applicable)
    if filename.startswith("RADA"):
        spart = filename.split("-")
        if len(spart) > 2:
            code = spart[1]
            radartype = spart[2]
    elif filename.startswith("Z"):
        spart = filename.split("_")
        if len(spart) > 7:
            code = spart[3]
            radartype = spart[7]
    return code, radartype


class StandardData(RadarBase):
    r"""
    Class reading data in standard format.

    Args:
        file (str, IO): Path points to the file or a file object.
    """
    # fmt: off
    dtype_corr = {1:'TREF', 2:'REF', 3:'VEL', 4:'SW', 5:'SQI', 6:'CPA', 7:'ZDR', 8:'LDR',
                  9:'RHO', 10:'PHI', 11:'KDP', 12:'CP', 14:'HCL', 15:'CF', 16:'SNRH',
                  17:'SNRV', 19:'POTS', 21:'COP', 26:'VELSZ', 27:'DR', 32:'Zc', 33:'Vc',
                  34:'Wc', 35:'ZDRc'}
    # fmt: on
    def __init__(self, file: Any):
        with prepare_file(file) as self.f:
            self._parse()
        self._update_radar_info()
        # In standard data, station information stored in file
        # has higher priority, so we override some information.
        self.stationlat = self.geo["lat"]
        self.stationlon = self.geo["lon"]
        self.radarheight = self.geo["height"]
        self.radarname = self.geo["name"]
        if self.name == "None":
            # Last resort to find info
            self.name = self.geo["name"]
        self.angleindex_r = self.available_tilt("REF")  # API consistency
        # del self.geo

    def _parse(self):
        header = np.frombuffer(self.f.read(32), SDD_header) #通用头块,32字节
        if header["magic_number"] != 0x4D545352:
            raise RadarDecodeError("Invalid standard data")
        site_config_dtype = SDD_site
        task_dtype = SDD_task
        cut_config_dtype = SDD_cut
        if header["generic_type"] == 16:
            site_config_dtype = PA_SDD_site
            task_dtype = PA_SDD_task
            cut_config_dtype = PA_SDD_cut
            header_length = 128
            radial_header_dtype = PA_SDD_rad_header
            self._is_phased_array = True
        else:
            header_length = 64
            radial_header_dtype = SDD_rad_header
            self._is_phased_array = False
        site_config = np.frombuffer(self.f.read(128), site_config_dtype) #站点配置块,128字节
        self.code = (
            site_config["site_code"][0]
            .decode("ascii", errors="ignore")
            .replace("\x00", "")
        )
        freq = site_config["frequency"][0]
        self.wavelength = 3e8 / freq / 10000
        self.geo = geo = dict()
        geo["lat"] = site_config["Latitude"][0]
        geo["lon"] = site_config["Longitude"][0]
        geo["height"] = site_config["ground_height"][0]
        geo["name"] = site_config["site_name"][0].decode("ascii", errors="ignore")
#--------------------------------------------------------------------------------------
        geo['Antenna_height'] = site_config["antenna_height"][0]
        geo['Ground_height'] = site_config["ground_height"][0]
        geo['Freq'] = site_config["frequency"][0]
        geo['Beam_Width_Hori'] = site_config["beam_width_hori"][0]
        geo['Beam_Width_Vert'] = site_config["beam_width_vert"][0]
        geo['Serv_Ver'] = site_config["RDA_version"][0]


        # 任务配置块 256字节
        task = np.frombuffer(self.f.read(256), task_dtype)
        self.task_name = (
            task["task_name"][0].decode("ascii", errors="ignore").split("\x00")[0]
        )
        epoch_seconds = datetime.timedelta(
            seconds=int(task["scan_start_time"][0])
        ).total_seconds()
#--------------------------------------------------------------------------------------
        geo['polar_type'] = task["polar_type"][0]
        geo['scan_type'] = task["scan_type"][0]
        geo['pulse_width'] = task["pulse_width"][0]
        geo['scan_start_time'] = task["scan_start_time"][0]
        geo['cut_number'] = task["cut_number"][0]
        # self.pol_type = task["polar_type"][0]
        # self.scan_type = task["scan_type"][0]
        # self.pulse_width = task["pulse_width"][0]
        # self.scan_ST = task["scan_start_time"][0]
        self.cut_num = task["cut_number"][0]


        self.scantime = epoch_seconds_to_utc(epoch_seconds)
        if self._is_phased_array:
            san_beam_number = task["san_beam_number"][0]
            self.pa_beam = np.frombuffer(
                self.f.read(san_beam_number * 640), PA_SDD_beam
            )


        #扫描配置块 256 * 扫描层数
        scan_config = np.frombuffer(self.f.read(256 * self.cut_num), cut_config_dtype)
        self.scan_config = list()
        scan_config_cls = ScanConfigPA if self._is_phased_array else ScanConfig
        for i in scan_config:
            _config = scan_config_cls(*i)
            if _config.dop_reso > 32768:  # fine detection scan(“精细探测扫描”) 多普勒分辨率（dop_reso）是否大于 32768
                true_reso = np.round_((_config.dop_reso - 32768) / 100, 1)
                _config_element = list(_config)
                if self._is_phased_array:
                    _config_element[21] = true_reso
                    _config_element[22] = true_reso
                else:
                    _config_element[11] = true_reso
                    _config_element[12] = true_reso
                _config = scan_config_cls(*_config_element)
            self.scan_config.append(_config)
        # TODO: improve repr
        data = dict() # 初始化数据存储字典，用于保存雷达原始数据（如反射率、速度等）
        aux = dict() # 初始化辅助信息字典，用于保存方位角、仰角、数据校准参数（比例和偏移量）等
        if task["scan_type"] == 2:  # 2表示单层RHI扫描模式
            self.scan_type = "RHI"
        else:
            # 实际还有其他扫描类型，但当前版本暂不支持，默认按PPI处理
            self.scan_type = "PPI"
        # Some attributes that are used only for converting to pyart.core.Radar instances
        self._time_radial = list() # 存储每个径向数据的时间戳（单位：秒）
        self._sweep_start_ray_index = list() # 存储每个扫描层（sweep）的起始径向索引
        self._sweep_end_ray_index = list() # 存储每个扫描层（sweep）的结束径向索引
        # 记录已读取的径向数量
        radial_count = 0

        while 1:
            try:
                header_bytes = self.f.read(header_length) # 64字节的径向头块
                if not header_bytes:
                    # Fix for single-tilt file
                    break
                radial_header = np.frombuffer(header_bytes, radial_header_dtype) #将字节流解析为结构化的径向头部数据（按预设的radial_header_dtype格式）
                if radial_header["zip_type"][0] == 1:  # LZO compression
                    raise NotImplementedError("LZO compressed file is not supported")
                self._time_radial.append( # 计算当前径向的时间戳
                    radial_header["seconds"][0] + radial_header["microseconds"][0] / 1e6
                )
                el_num = radial_header["elevation_number"][0] - 1 # 获取当前径向所属的仰角层编号
                if el_num not in data.keys(): # 若当前仰角层是首次出现，初始化该层的数据存储结构
                    data[el_num] = defaultdict(list)
                    aux[el_num] = defaultdict(list)
                # 存储当前径向的方位角到辅助信息中（按仰角层分类）
                aux[el_num]["azimuth"].append(radial_header["azimuth"][0])
                aux[el_num]["elevation"].append(radial_header["elevation"][0])
#---------------------------------------------------------------------------------------------
                aux[el_num]["radial_state"].append(radial_header["radial_state"][0])
                aux[el_num]["seq_num"].append(radial_header["seq_number"][0])
                aux[el_num]["rad_num"].append(radial_header["radial_number"][0])
                aux[el_num]["Ele_Num"].append(radial_header["elevation_number"][0])
                aux[el_num]["Sec"].append(radial_header["seconds"][0])
                aux[el_num]["Mic_Sec"].append(radial_header["microseconds"][0])
                aux[el_num]["Moment_um"].append(radial_header["moment_number"][0])

                for _ in range(radial_header["moment_number"][0]):
                    # 读取32字节的数据头块
                    moment_header = np.frombuffer(self.f.read(32), SDD_mom_header)
                    if moment_header["block_length"][0] == 0:
                        # Some ill-formed files
                        continue
                    dtype_code = moment_header["data_type"][0]

                    dtype = self.dtype_corr.get(dtype_code, None)
                    data_body = np.frombuffer(
                        self.f.read(moment_header["block_length"][0]),
                        "u{}".format(moment_header["bin_length"][0]),
                    )
                    if not dtype:
                        warnings.warn(
                            "Data type {} not understood, skipping".format(dtype_code),
                            RuntimeWarning,
                        )
                        continue
                    # 若当前仰角层首次出现该数据类型，记录其校准参数（比例和偏移量）
                    if dtype not in aux[el_num].keys():
                        scale = moment_header["scale"][0]
                        offset = moment_header["offset"][0]
                        aux[el_num][dtype] = (scale, offset)
                    # 后续计算实际物理量时使用：实际值 = 原始值 * scale + offset
                    # 将原始数据存入对应的数据结构（暂不进行校准计算，校准延迟到get_raw方法）
                    data[el_num][dtype].append(data_body)
                radial_state = radial_header["radial_state"][0]

                if radial_state in [0, 3]:
                    # Start of tilt or volume scan
                    self._sweep_start_ray_index.append(radial_count)
                elif radial_state in [2, 4]:
                    self._sweep_end_ray_index.append(radial_count)
                radial_count += 1
                if radial_state in [4, 6]:  # End scan
                    break
            except EOFError:
                # Decode broken files as much as possible
                warnings.warn("Broken compressed file detected.", RuntimeWarning)
                break

        self.data = data
        self.aux = aux
        self.el = [i.elev for i in self.scan_config]

    @classmethod
    def merge(cls, files: List[str], output: str):
        r"""
        Merge single-tilt standard data into a volumetric scan

        Args:
            files (List[str]): List of path of data to be merged

            output (str): The file path to store the merged data
        """
        with prepare_file(files[0]) as first_file:
            first_file.seek(160)
            task = np.frombuffer(first_file.read(256), SDD_task)
            cut_num = task["cut_number"][0]
            total_seek_bytes = first_file.tell() + 256 * cut_num
            all_tilt_data = [b""] * cut_num
            first_file.seek(0)
            header_bytes = first_file.read(total_seek_bytes)
            rad = np.frombuffer(first_file.read(64), SDD_rad_header)
            el_num = rad["elevation_number"][0] - 1
            first_file.seek(total_seek_bytes)
            all_tilt_data[el_num] = first_file.read()
            for f in files[1:]:
                with prepare_file(f) as buf:
                    buf.seek(total_seek_bytes)
                    rad = np.frombuffer(buf.read(64), SDD_rad_header)
                    buf.seek(total_seek_bytes)
                    el_num = rad["elevation_number"][0] - 1
                    all_tilt_data[el_num] = buf.read()
        with open(output, "wb") as out:
            out.write(header_bytes)
            out.write(b"".join(all_tilt_data))

    def get_raw(
        self, tilt: int, drange: Number_T, dtype: str
    ) -> Union[np.ndarray, tuple]:
        r"""
        Get radar raw data

        Args:
            tilt (int): Index of elevation angle starting from zero.

            drange (float): Radius of data.

            dtype (str): Type of product (REF, VEL, etc.)

        Returns:
            numpy.ndarray or tuple of numpy.ndarray: Raw data
        """
        # The scan number is set to zero in RHI mode.
        self.tilt = tilt if self.scan_type == "PPI" else 0
        if tilt not in self.data:
            raise RadarDecodeError("Tilt {} does not exist.".format(tilt))
        if dtype not in self.data[tilt]:
            raise RadarDecodeError(
                "Product {} does not exist in tilt {}".format(dtype, tilt)
            )
        if self.scan_type == "RHI":
            max_range = self.scan_config[0].max_range1 / 1000
            if drange > max_range:
                drange = max_range
        self.drange = drange
        self.elev = self.el[tilt]
        if dtype in ["VEL", "SW", "VELSZ"]:
            reso = self.scan_config[tilt].dop_reso / 1000
        else:
            reso = self.scan_config[tilt].log_reso / 1000
        raw = np.array(self.data[tilt][dtype])
        ngates = int(drange // reso)
        if raw.size == 0:
            warnings.warn("Empty data", RuntimeWarning)
            # Calculate size equivalent
            nrays = len(self.aux[tilt]["azimuth"])
            out = np.zeros((nrays, ngates)) * np.ma.masked
            return out
        # Data below 5 are used as reserved codes, which are used to indicate other
        # information instead of real data, so they should be masked.
        data = np.ma.masked_less(raw, 5)
        cut = data[:, :ngates]
        shape_diff = ngates - cut.shape[1]
        append = np.zeros((cut.shape[0], int(shape_diff))) * np.ma.masked
        if dtype in ["VEL", "SW", "VELSZ"]:
            # The reserved code 1 indicates folded velocity.
            # These region will be shaded by color of `RF`.
            rf = np.ma.masked_not_equal(cut.data, 1)
            rf = np.ma.hstack([rf, append])
        cut = np.ma.hstack([cut, append])
        scale, offset = self.aux[tilt][dtype]
        r = (cut - offset) / scale
        if dtype in ["VEL", "SW", "VELSZ"]:
            ret = (r, rf)
            # RF data is separately packed into the data.
        else:
            ret = r
        return ret



    def get_data_detail(self, tilt: int, drange: Number_T, dtype: str) -> xr.Dataset:
        r"""
        Get radar data with extra information

        Args:
            tilt (int): Index of elevation angle starting from zero.

            drange (float): Radius of data.

            dtype (str): Type of product (REF, VEL, etc.)

        Returns:
            xarray.Dataset: Data.
        """
        ret = self.get_raw(tilt, drange, dtype)
        if dtype in ["VEL", "SW", "VELSZ"]:
            reso = self.scan_config[tilt].dop_reso / 1000
        else:
            reso = self.scan_config[tilt].log_reso / 1000
        shape = ret[0].shape[1] if isinstance(ret, tuple) else ret.shape[1]
        attr_list = {"PRF_1":"PRF1",
                     "PRF_2":"PRF2",
                     "Deal_mode":"dealias_mode",
                     "AZI":"azimuth",
                     "ELE":"elev",
                     "St_Angle":"start_angle",
                     "End_Angle":"end_angle",
                     "Ang_Res":"angular_reso",
                     "Scan_Spd":"scan_spd",
                     "Log_Res":"log_reso",
                     "Dop_Res":"dop_reso",
                     "Max_Range1":"max_range1",
                     "Max_Range2":"max_range2",
                     "St_Range":"start_range",
                     "Sample1":"sample1",
                     "Sample2":"sample2",
                     "At_Loss":"atmos_loss",
                     "Nyq_Spd":"nyquist_spd",
                     "Dir":"direction",
                     "Cla_type":"ground_clutter_classifier_type",
                     "Fil_type":"ground_clutter_filter_type",
                     "Fil_not_width":"ground_clutter_filter_notch_width",
                     "Fil_win":"ground_clutter_filter_window"}
        scan_config_dict = {}
        for key,name in attr_list.items():
            scan_config_dict[key] = []
        # for config in self.scan_config:
        config_dict = self.scan_config[tilt]._asdict()
        for key, name in attr_list.items():
            scan_config_dict[key].append(config_dict[name])

        radial_state = self.aux[self.tilt]["radial_state"]
        seq_num = self.aux[self.tilt]["seq_num"]
        rad_num = self.aux[self.tilt]["rad_num"]
        Ele_Num = self.aux[self.tilt]["Ele_Num"]
        Sec = self.aux[self.tilt]["Sec"]
        Mic_Sec = self.aux[self.tilt]["Mic_Sec"]
        Moment_um = self.aux[self.tilt]["Moment_um"]
        if self.scan_type == "PPI":
            x, y, z, d, a = self.projection(reso)
            if dtype in ["VEL", "SW", "VELSZ"]:
                da = xr.DataArray(ret[0], coords=[a, d], dims=["azimuth", "distance"])
            else:
                da = xr.DataArray(ret, coords=[a, d], dims=["azimuth", "distance"])
            da["radial_state"] = xr.DataArray(radial_state, coords={"azimuth": a}, dims=["azimuth"])
            da["seq_num"] = xr.DataArray(seq_num, coords={"azimuth": a}, dims=["azimuth"])
            da["rad_num"] = xr.DataArray(rad_num, coords={"azimuth": a}, dims=["azimuth"])
            da["Ele_Num"] = xr.DataArray(Ele_Num, coords={"azimuth": a}, dims=["azimuth"])
            da["Sec"] = xr.DataArray(Sec, coords={"azimuth": a}, dims=["azimuth"])
            da["Mic_Sec"] = xr.DataArray(Mic_Sec, coords={"azimuth": a}, dims=["azimuth"])
            da["Moment_um"] = xr.DataArray(Moment_um, coords={"azimuth": a}, dims=["azimuth"])
            attr = {
                    "site_code": self.code,
                    "site_name": self.name,
                    "site_latitude": self.stationlat,
                    "site_longitude": self.stationlon,
                    'Antenna_height':self.geo['Antenna_height'],
                    'Ground_height':self.geo['Ground_height'],
                    'Freq':self.geo['Freq'],
                    'Beam_Width_Hori':self.geo['Beam_Width_Hori'],
                    'Beam_Width_Vert':self.geo['Beam_Width_Vert'],
                    'Serv_Ver':self.geo['Serv_Ver'],
                    "elevation": self.elev,
                    "range": int(np.round(shape * reso)),
                    "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
                    "tangential_reso": reso,
                    "nyquist_vel": self.scan_config[tilt].nyquist_spd,
                    "task": self.task_name,
                    "polar_type":self.geo['polar_type'],
                    "scan_type":self.geo['scan_type'],
                    "pulse_width":self.geo['pulse_width'],
                    "scan_start_time":self.geo['scan_start_time'],
                    "cut_number":self.geo['cut_number'],
                }
            attr.update(scan_config_dict)
            ds = xr.Dataset(
                {dtype: da},
                attrs=attr
            )
            # ds["longitude"] = (["azimuth", "distance"], x)
            # ds["latitude"] = (["azimuth", "distance"], y)
            # ds["height"] = (["azimuth", "distance"], z)
            if dtype in ["VEL", "SW", "VELSZ"]:
                ds["RF"] = (["azimuth", "distance"], ret[1])
        else:
            warnings.warn(
                f"当前扫描模式为 {self.scan_type}（非PPI模式），将使用手动投影逻辑。",
                UserWarning  # 警告类型（可根据需要调整）
            )
            return None
            # Manual projection
            gate_num = ret[0].shape[1] if dtype in ["VEL", "SW"] else ret.shape[1]
            dist = np.linspace(reso, self.drange, gate_num)
            azimuth = self.aux[tilt]["azimuth"][0]
            elev = self.aux[tilt]["elevation"]
            d, e = np.meshgrid(dist, elev)
            h = height(d, e, self.radarheight)
            if dtype in ["VEL", "SW", "VELSZ"]:
                da = xr.DataArray(
                    ret[0], coords=[elev, dist], dims=["tilt", "distance"]
                )
            else:
                da = xr.DataArray(ret, coords=[elev, dist], dims=["tilt", "distance"])

            # Calculate the "start" and "end" of RHI scan
            # to facilitate tick labeling
            start_lon = self.stationlon
            start_lat = self.stationlat
            end_lon, end_lat = get_coordinate(
                self.drange, np.deg2rad(azimuth), 0, self.stationlon, self.stationlat
            )

            ds = xr.Dataset(
                {dtype: da},
                attrs={
                    "range": self.drange,
                    "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
                    "site_code": self.code,
                    "site_name": self.name,
                    "site_longitude": self.stationlon,
                    "site_latitude": self.stationlat,
                    "tangential_reso": reso,
                    "azimuth": azimuth,
                    "start_lon": start_lon,
                    "start_lat": start_lat,
                    "end_lon": end_lon,
                    "end_lat": end_lat,
                    "nyquist_vel": self.scan_config[tilt].nyquist_spd,
                },
            )
            ds["x_cor"] = (["tilt", "distance"], d)
            ds["y_cor"] = (["tilt", "distance"], h)
        return ds

    def projection(self, reso: float) -> tuple:
        r = self.get_range(self.drange, reso)
        theta = np.deg2rad(self.aux[self.tilt]["azimuth"])

        lonx, latx = get_coordinate(
            r, theta, self.elev, self.stationlon, self.stationlat
        )
        hght = (
            height(r, self.elev, self.radarheight)
            * np.ones(theta.shape[0])[:, np.newaxis]
        )


        return lonx, latx, hght, r, theta

    def available_tilt(self, product: str) -> List[int]:
        r"""Get all available tilts for given product"""
        tilt = list()
        for i in list(self.data.keys()):
            if product in self.data[i].keys():
                tilt.append(i)
        return tilt

    def get_data_multi(self, tilt: int, drange: Number_T, dtype) -> xr.Dataset:
        r"""
        Get radar data with extra information, fixing distance dimension mismatch across products.

        Args:
            tilt (int): Index of elevation angle starting from zero.
            drange (float): Radius of data.
            dtype (str or list): Type of product (REF, VEL, etc.) or list of product types.

        Returns:
            xarray.Dataset: Data with consistent dimensions.
        """
        if isinstance(dtype, str):
            dtype_list = [dtype]
        else:
            dtype_list = dtype

        # 获取第一个产品的数据来建立基础结构
        first_dtype = dtype_list[0]
        ret = self.get_raw(tilt, drange, first_dtype)
        # 统一处理输入类型为列表
        if isinstance(dtype, str):
            dtype_list = [dtype]
        else:
            dtype_list = dtype

        # 1. 收集所有产品的分辨率和对应的距离门数量
        product_info = []
        for dt in dtype_list:
            if dt in ["VEL", "SW", "VELSZ"]:
                reso = self.scan_config[tilt].dop_reso / 1000
            else:
                reso = self.scan_config[tilt].log_reso / 1000
            ngates = int(drange // reso)
            product_info.append({"dtype": dt, "reso": reso, "ngates": ngates})

        # 2. 确定统一的距离门数量（取最小值，保证所有产品数据完整）
        min_ngates = min(info["ngates"] for info in product_info)  # 关键修改：用min替代max
        # 基于最小ngates计算统一分辨率（此时范围由最小ngates决定，更大更完整）
        unified_reso = drange / min_ngates if min_ngates != 0 else 0

        # 3. 获取方位角坐标和统一的距离坐标（长度=min_ngates）
        if self.scan_type == "PPI":
            _, _, _, d, a = self.projection(unified_reso)
            # 确保distance坐标长度与最小ngates一致（截断到最小长度）
            d = d[:min_ngates]  # 因为min_ngates是最小的，所以d的长度肯定≥min_ngates
        else:
            warnings.warn(f"当前扫描模式为 {self.scan_type}（非PPI模式），暂不支持多产品合并。", UserWarning)
            return None

        # 4. 准备数据变量字典
        data_vars = {}
        # 获取辅助数据
        radial_state = self.aux[self.tilt]["radial_state"]
        seq_num = self.aux[self.tilt]["seq_num"]
        rad_num = self.aux[self.tilt]["rad_num"]
        Ele_Num = self.aux[self.tilt]["Ele_Num"]
        Sec = self.aux[self.tilt]["Sec"]
        Mic_Sec = self.aux[self.tilt]["Mic_Sec"]
        Moment_um = self.aux[self.tilt]["Moment_um"]

        # 5. 处理每个产品的数据（统一截断到最小ngates，无需填充）
        for info in product_info:
            dt = info["dtype"]

            # 获取原始数据
            ret_data = self.get_raw(tilt, drange, dt)

            # 处理速度类产品
            if dt in ["VEL", "SW", "VELSZ"]:
                data = ret_data[0]
                rf_data = ret_data[1]

                # 截断到最小ngates（所有产品的ngates≥min_ngates，因此直接截断即可）
                data = data[:, :min_ngates]
                rf_data = rf_data[:, :min_ngates]

                data_vars[dt] = xr.DataArray(data, coords=[a, d], dims=["azimuth", "distance"])
                data_vars[f"{dt}_RF"] = xr.DataArray(rf_data, coords=[a, d], dims=["azimuth", "distance"])

            # 处理其他产品
            else:
                data = ret_data
                # 截断到最小ngates
                data = data[:, :min_ngates]
                data_vars[dt] = xr.DataArray(data, coords=[a, d], dims=["azimuth", "distance"])

        # 6. 添加辅助数据变量
        coord_vars = {
            "radial_state": xr.DataArray(radial_state, coords={"azimuth": a}, dims=["azimuth"]),
            "seq_num": xr.DataArray(seq_num, coords={"azimuth": a}, dims=["azimuth"]),
            "rad_num": xr.DataArray(rad_num, coords={"azimuth": a}, dims=["azimuth"]),
            "Ele_Num": xr.DataArray(Ele_Num, coords={"azimuth": a}, dims=["azimuth"]),
            "Sec": xr.DataArray(Sec, coords={"azimuth": a}, dims=["azimuth"]),
            "Mic_Sec": xr.DataArray(Mic_Sec, coords={"azimuth": a}, dims=["azimuth"]),
            "Moment_um": xr.DataArray(Moment_um, coords={"azimuth": a}, dims=["azimuth"]),
        }
        data_vars.update(coord_vars)

        # 7. 设置数据集属性（更新与ngates相关的描述）
        attr = {
            "site_code": self.code,
            "site_name": self.name,
            "site_latitude": self.stationlat,
            "site_longitude": self.stationlon,
            'Antenna_height': self.geo['Antenna_height'],
            'Ground_height': self.geo['Ground_height'],
            'Freq': self.geo['Freq'],
            'Beam_Width_Hori': self.geo['Beam_Width_Hori'],
            'Beam_Width_Vert': self.geo['Beam_Width_Vert'],
            'Serv_Ver': self.geo['Serv_Ver'],
            # "elevation": self.elev,
            # "range": int(np.round(min_ngates * unified_reso)),  # 范围由最小ngates决定
            # "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
            # "unified_tangential_reso": unified_reso,
            # "original_resolutions": {info["dtype"]: info["reso"] for info in product_info},
            # "min_ngates_used": min_ngates,  # 记录使用的最小距离门数量
            # "nyquist_vel": self.scan_config[tilt].nyquist_spd if "VEL" in dtype_list else None,
            "task_name": self.task_name,
            "polar_type": self.geo['polar_type'],
            "scan_type": self.geo['scan_type'],
            "pulse_width": self.geo['pulse_width'],
            "scan_start_time": self.geo['scan_start_time'],
            "cut_number": self.geo['cut_number'],
        }

        return xr.Dataset(data_vars, attrs=attr)

    def get_multi_elevation_data_old(self, elevation_config: Dict[int, List[str]], drange: float) -> xr.Dataset:
        """
        获取多个仰角的雷达数据并合并到一个数据集中（含仰角去重，保留所有distance信息）
        """
        attr_list = {"PRF_1": "PRF1",
                     "PRF_2": "PRF2",
                     "Deal_mode": "dealias_mode",
                     "AZI": "azimuth",
                     "ELE": "elev",
                     "St_Angle": "start_angle",
                     "End_Angle": "end_angle",
                     "Ang_Res": "angular_reso",
                     "Scan_Spd": "scan_spd",
                     "Log_Res": "log_reso",
                     "Dop_Res": "dop_reso",
                     "Max_Range1": "max_range1",
                     "Max_Range2": "max_range2",
                     "St_Range": "start_range",
                     "Sample1": "sample1",
                     "Sample2": "sample2",
                     "At_Loss": "atmos_loss",
                     "Nyq_Spd": "nyquist_spd",
                     "Dir": "direction",
                     "Cla_type": "ground_clutter_classifier_type",
                     "Fil_type": "ground_clutter_filter_type",
                     "Fil_not_width": "ground_clutter_filter_notch_width",
                     "Fil_win": "ground_clutter_filter_window"
                     }

        # 收集所有产品类型
        all_products = set()
        for products in elevation_config.values():
            all_products.update(products)
        all_products = sorted(list(all_products))

        # 收集所有仰角的数据（带去重逻辑）
        elevation_datasets = {}  # 原始tilt索引: 数据集
        raw_elevation_angles = []  # 原始仰角值（可能重复）
        raw_tilt_indices = []  # 原始tilt索引（与原始仰角对应）
        all_azimuths = set()
        all_distances = set()  # 新增：收集所有distance值

        # 第一步：收集所有原始数据（含重复仰角）
        scan_config_dict = {key: [] for key in attr_list.keys()}  # 原始配置（含重复）
        for tilt_idx, products in elevation_config.items():
            # 获取该tilt的配置
            config_dict = self.scan_config[tilt_idx]._asdict()

            # 记录原始仰角和tilt索引
            elev_value = config_dict['elev']
            raw_elevation_angles.append(elev_value)
            raw_tilt_indices.append(tilt_idx)
            # 记录配置信息（原始顺序，含重复）
            for key, name in attr_list.items():
                scan_config_dict[key].append(config_dict[name])
            # 加载数据集
            ds = self.get_data_multi(tilt_idx, drange, products)
            if ds is not None:
                elevation_datasets[tilt_idx] = ds
                # 收集方位角和距离信息
                azimuth_values = ds.azimuth.values
                all_azimuths.update(azimuth_values)
                distance_values = ds.distance.values  # 新增：收集当前数据集的distance
                all_distances.update(distance_values)

        if not elevation_datasets:
            print("elevation_datasets",elevation_datasets)
            return None

        # 第二步：对仰角去重，记录映射关系（原始索引 → 去重后索引）
        unique_elevations = []
        unique_indices = []  # 去重后保留的原始tilt索引
        seen = set()
        for idx, (elev, tilt_idx) in enumerate(zip(raw_elevation_angles, raw_tilt_indices)):
            if elev not in seen:
                seen.add(elev)
                unique_elevations.append(elev)
                unique_indices.append(idx)  # 记录原始配置的索引

        # 第三步：根据去重后的索引调整配置信息
        unique_scan_config = {key: [] for key in attr_list.keys()}
        for key in scan_config_dict:
            unique_scan_config[key] = [scan_config_dict[key][i] for i in unique_indices]

        # 创建坐标系统（使用所有数据集的distance并集）
        unified_azimuth = np.array(sorted(all_azimuths))
        unified_distance = np.array(sorted(all_distances))  # 合并所有distance并排序
        elevation_coord = np.array(unique_elevations)  # 去重后的仰角坐标
        n_unique_elev = len(unique_elevations)

        # 第四步：为每个产品创建3D数组（包含所有distance信息）
        data_vars = {}
        for product in all_products:
            # 数组形状：(去重后的仰角数, 方位角数, 合并后的距离数)
            product_data = np.full((n_unique_elev, len(unified_azimuth), len(unified_distance)),
                                   np.nan, dtype=np.float32)

            # 遍历去重后的仰角，填充数据
            for unique_idx, orig_idx in enumerate(unique_indices):
                tilt_idx = raw_tilt_indices[orig_idx]  # 原始tilt索引
                ds = elevation_datasets.get(tilt_idx)
                if ds is None or product not in ds.data_vars:
                    continue  # 跳过无数据的情况

                # 获取该仰角的数据和坐标
                azimuth_values = ds.azimuth.values
                distance_values = ds.distance.values  # 当前数据集的distance
                product_values = ds[product].values

                # 映射到统一坐标（精确匹配distance值）
                for i, az in enumerate(azimuth_values):
                    # 找到方位角在统一坐标中的位置
                    az_idx = np.where(np.isclose(unified_azimuth, az, atol=1e-4))[0]
                    if len(az_idx) == 0 or i >= product_values.shape[0]:
                        continue
                    az_idx = az_idx[0]

                    # 匹配距离坐标（关键修改：逐点匹配所有distance）
                    for j, dist in enumerate(distance_values):
                        dist_idx = np.where(np.isclose(unified_distance, dist, atol=1e-4))[0]
                        if len(dist_idx) > 0 and j < product_values.shape[1]:
                            product_data[unique_idx, az_idx, dist_idx[0]] = product_values[i, j]

            data_vars[product] = (["elevation", "azimuth", "distance"], product_data)

        # 第五步：处理辅助变量（同步去重后的维度）
        auxiliary_vars = {}
        aux_fields = ["radial_state", "seq_num", "rad_num", "Ele_Num", "Sec", "Mic_Sec", "Moment_um"]
        for aux_field in aux_fields:
            # 检查第一个数据集是否有该辅助字段
            first_ds = list(elevation_datasets.values())[0]
            if aux_field not in first_ds.coords:
                continue

            # 辅助变量形状：(去重后的仰角数, 方位角数)
            aux_3d = np.full((n_unique_elev, len(unified_azimuth)), np.nan, dtype=np.float32)

            for unique_idx, orig_idx in enumerate(unique_indices):
                tilt_idx = raw_tilt_indices[orig_idx]
                ds = elevation_datasets.get(tilt_idx)
                if ds is None or aux_field not in ds.coords:
                    continue

                azimuth_values = ds.azimuth.values
                # 统一方位角格式（角度制）
                # if np.max(azimuth_values) <= 2 * np.pi:
                #     azimuth_degrees = np.rad2deg(azimuth_values)
                # else:
                #     azimuth_degrees = azimuth_values
                azimuth_degrees = azimuth_values
                aux_values = ds[aux_field].values

                # 映射到统一方位角
                for i, az in enumerate(azimuth_degrees):
                    az_idx = np.where(np.isclose(unified_azimuth, az, atol=1e-4))[0]
                    if len(az_idx) > 0 and i < len(aux_values):
                        aux_3d[unique_idx, az_idx[0]] = aux_values[i]

            auxiliary_vars[aux_field] = (["elevation", "azimuth"], aux_3d)

        # 合并所有数据变量
        all_data_vars = {
            **data_vars, **{key: (["elevation"], unique_scan_config[key]) for key in unique_scan_config},
            **auxiliary_vars
        }

        # 创建综合属性
        combined_attrs = dict(list(elevation_datasets.values())[0].attrs)
        # combined_attrs.update({
        #     "elevation_angles": unique_elevations,
        #     "elevation_count": n_unique_elev,
        #     "products": list(all_products),
        #     "azimuth_count": len(unified_azimuth),
        #     "distance_count": len(unified_distance),  # 记录总距离点数
        #     "max_distance": float(unified_distance[-1]) if len(unified_distance) > 0 else 0,
        # })
        self.scan_config_dict = unique_scan_config

        # 创建坐标
        coords = {
            "elevation": elevation_coord,
            "azimuth": unified_azimuth,
            "distance": unified_distance  # 使用合并后的distance坐标
        }

        # 创建最终的数据集
        combined_ds = xr.Dataset(all_data_vars, coords=coords, attrs=combined_attrs)

        return combined_ds

    def get_multi_elevation_data(self, elevation_config: Dict[int, List[str]], drange: float) -> xr.Dataset:
        """
        获取多个仰角的雷达数据并合并到一个数据集中（含仰角去重，保留所有distance信息）
        优化点：向量化解法替代嵌套循环，预计算索引映射加速查找
        """
        attr_list = {
            "PRF_1": "PRF1", "PRF_2": "PRF2", "Deal_mode": "dealias_mode",
            "AZI": "azimuth", "ELE": "elev", "St_Angle": "start_angle",
            "End_Angle": "end_angle", "Ang_Res": "angular_reso", "Scan_Spd": "scan_spd",
            "Log_Res": "log_reso", "Dop_Res": "dop_reso", "Max_Range1": "max_range1",
            "Max_Range2": "max_range2", "St_Range": "start_range", "Sample1": "sample1",
            "Sample2": "sample2", "At_Loss": "atmos_loss", "Nyq_Spd": "nyquist_spd",
            "Dir": "direction", "Cla_type": "ground_clutter_classifier_type",
            "Fil_type": "ground_clutter_filter_type", "Fil_not_width": "ground_clutter_filter_notch_width",
            "Fil_win": "ground_clutter_filter_window"
        }

        # 收集所有产品类型（无优化，保持原样）
        all_products = set()
        for products in elevation_config.values():
            all_products.update(products)
        all_products = sorted(list(all_products))

        # 第一步：收集所有原始数据（含重复仰角）
        elevation_datasets = {}  # 原始tilt索引: 数据集
        raw_elevation_angles = []  # 原始仰角值（可能重复）
        raw_tilt_indices = []  # 原始tilt索引
        all_azimuths = []  # 用列表替代set，后续一次性去重排序（减少set插入开销）
        all_distances = []  # 用列表替代set

        scan_config_dict = {key: [] for key in attr_list.keys()}
        for tilt_idx, products in elevation_config.items():
            config_dict = self.scan_config[tilt_idx]._asdict()
            elev_value = config_dict['elev']
            raw_elevation_angles.append(elev_value)
            raw_tilt_indices.append(tilt_idx)

            # 记录配置信息
            for key, name in attr_list.items():
                scan_config_dict[key].append(config_dict[name])

            # 加载数据集
            ds = self.get_data_multi(tilt_idx, drange, products)
            if ds is not None:
                elevation_datasets[tilt_idx] = ds
                # 批量收集方位角和距离（后续统一去重）
                all_azimuths.extend(ds.azimuth.values)
                all_distances.extend(ds.distance.values)

        if not elevation_datasets:
            print("elevation_datasets is empty")
            return None

        # 统一处理方位角和距离的去重与排序（向量化操作）
        unified_azimuth = np.unique(all_azimuths)  # 四舍五入后去重（减少精度干扰）
        unified_distance = np.unique(all_distances)  # 四舍五入到5位小数，平衡精度与效率
        # 预创建坐标到索引的映射表（O(1)查找）
        az_to_idx = {az: idx for idx, az in enumerate(unified_azimuth)}
        dist_to_idx = {dist: idx for idx, dist in enumerate(unified_distance)}

        # 第二步：仰角去重（用pandas加速去重，保留原始顺序）
        import pandas as pd  # 若已有导入可移除
        # 用DataFrame去重并保留首次出现的索引
        df_elev = pd.DataFrame({
            'elev': raw_elevation_angles,
            'orig_idx': range(len(raw_elevation_angles)),
            'tilt_idx': raw_tilt_indices
        })
        df_unique_elev = df_elev.drop_duplicates(subset='elev', keep='first').sort_index()
        unique_elevations = df_unique_elev['elev'].values
        unique_indices = df_unique_elev['orig_idx'].values  # 去重后保留的原始配置索引
        unique_tilt_indices = df_unique_elev['tilt_idx'].values  # 去重后对应的tilt索引
        n_unique_elev = len(unique_elevations)

        # 第三步：调整配置信息（向量化索引）
        unique_scan_config = {
            key: [scan_config_dict[key][i] for i in unique_indices]
            for key in scan_config_dict
        }

        # 第四步：填充产品数据（核心优化：向量化解法）
        data_vars = {}
        for product in all_products:
            # 初始化产品数据数组（shape: [n_elev, n_az, n_dist]）
            product_data = np.full(
                (n_unique_elev, len(unified_azimuth), len(unified_distance)),
                np.nan, dtype=np.float32
            )

            # 遍历去重后的仰角（仅一层循环）
            for unique_idx, tilt_idx in enumerate(unique_tilt_indices):
                ds = elevation_datasets.get(tilt_idx)
                if ds is None or product not in ds.data_vars:
                    continue

                # 获取当前数据集的坐标和数据（四舍五入匹配映射表）
                az_values = ds.azimuth.values
                dist_values = ds.distance.values
                product_values = ds[product].values  # shape: [n_az_current, n_dist_current]

                # 批量查找有效坐标的索引（向量化操作，替代嵌套循环）
                # 1. 筛选在统一坐标中存在的方位角和距离
                valid_az_mask = np.isin(az_values, unified_azimuth)
                valid_dist_mask = np.isin(dist_values, unified_distance)
                valid_az_indices = np.where(valid_az_mask)[0]
                valid_dist_indices = np.where(valid_dist_mask)[0]

                if len(valid_az_indices) == 0 or len(valid_dist_indices) == 0:
                    print("无有效数据，跳过")
                    continue  # 无有效数据，跳过

                # 2. 映射到统一坐标的索引（利用预创建的字典）
                az_global_indices = np.array([az_to_idx[az] for az in az_values[valid_az_indices]])
                dist_global_indices = [dist_to_idx[dist] for dist in dist_values[valid_dist_indices]]

                # 3. 批量填充数据（用数组切片替代逐点赋值）
                # 提取有效数据块（当前数据集的有效区域）
                valid_data = product_values[valid_az_indices[:, np.newaxis], valid_dist_indices]
                # 填充到全局数组的对应位置
                product_data[unique_idx, az_global_indices[:, np.newaxis], dist_global_indices] = valid_data

            data_vars[product] = (["elevation", "azimuth", "distance"], product_data)

        # 第五步：处理辅助变量（同样使用向量化解法）
        auxiliary_vars = {}
        # 辅助变量列表（从返回的数据集可知这些变量存在于data_vars中）
        aux_fields = ["radial_state", "seq_num", "rad_num", "Ele_Num", "Sec", "Mic_Sec", "Moment_um"]

        # 为整数类型的辅助变量指定合适的填充值（避免使用NaN）
        aux_fill_values = {
            "radial_state": -9999,
            "seq_num": -9999,
            "rad_num": -9999,
            "Ele_Num": -9999,
            "Sec": -9999,
            "Mic_Sec": -9999,
            "Moment_um": -9999
        }

        for aux_field in aux_fields:
            # 1. 检查变量是否存在于第一个数据集的data_vars中（关键修正）
            first_ds = list(elevation_datasets.values())[0]
            # print(first_ds.data_vars)
            if aux_field not in first_ds.data_vars:
                print(f"辅助变量 {aux_field} 不在数据集中，跳过")
                continue

            # 2. 初始化辅助变量数组（使用int32类型和对应填充值）
            # 维度为 (elevation, azimuth)，与需求一致
            aux_data = np.full(
                (n_unique_elev, len(unified_azimuth)),
                aux_fill_values[aux_field],
                dtype=np.int32  # 所有辅助变量均为int32类型
            )

            # 3. 遍历每个去重后的仰角，填充数据
            for unique_idx, tilt_idx in enumerate(unique_tilt_indices):
                ds = elevation_datasets.get(tilt_idx)
                if ds is None or aux_field not in ds.data_vars:
                    continue  # 跳过无数据的情况

                # 获取当前数据集的方位角和辅助变量值
                az_values = ds.azimuth.values
                aux_values = ds[aux_field].values  # 从data_vars中获取（关键修正）

                # 验证辅助变量维度是否正确（应为(azimuth,)）
                if aux_values.shape != (len(az_values),):
                    print(f"辅助变量 {aux_field} 维度错误，预期{(len(az_values),)}，实际{aux_values.shape}")
                    continue

                # 4. 匹配全局方位角索引（向量化解法）
                # 查找当前方位角在全局方位角中的位置
                valid_az_mask = np.isin(az_values, unified_azimuth)
                valid_az_indices = np.where(valid_az_mask)[0]
                if len(valid_az_indices) == 0:
                    print(f"仰角 {unique_idx} 无有效方位角数据，跳过 {aux_field}")
                    continue

                # 映射到全局索引
                az_global_indices = [az_to_idx[az] for az in az_values[valid_az_indices]]

                # 5. 填充数据到对应位置
                aux_data[unique_idx, az_global_indices] = aux_values[valid_az_indices]

            # 6. 检查是否有有效数据，避免全填充值的变量
            if np.all(aux_data == aux_fill_values[aux_field]):
                print(f"辅助变量 {aux_field} 无有效数据，未添加")
                continue

            # 添加到辅助变量字典，维度为(elevation, azimuth)
            auxiliary_vars[aux_field] = (["elevation", "azimuth"], aux_data)
            # print(f"成功添加辅助变量 {aux_field}，维度: {aux_data.shape}")

        # 合并所有数据变量（保持原样）
        all_data_vars = {
            **data_vars, **{key: (["elevation"], unique_scan_config[key]) for key in unique_scan_config},
            **auxiliary_vars
        }
        # print(all_data_vars)

        # 创建综合属性和坐标（保持原样）
        combined_attrs = dict(list(elevation_datasets.values())[0].attrs)
        combined_attrs.update({
            # "elevation_angles": unique_elevations,
            # "elevation_count": n_unique_elev,
            "products": list(all_products),
            # "azimuth_count": len(unified_azimuth),
            # "distance_count": len(unified_distance),
            # "max_distance": float(unified_distance[-1]) if len(unified_distance) > 0 else 0,
        })
        self.scan_config_dict = unique_scan_config

        coords = {
            "elevation": unique_elevations,
            "azimuth": unified_azimuth,
            "distance": unified_distance
        }
        combined_ds = xr.Dataset(all_data_vars, coords=coords, attrs=combined_attrs)
        return combined_ds

    def save_multi_time_volume_chunked(self,radar_objects_list, elevation_configs, drange, out_file):
        """
        第二步：基于 metadata 分块写入 NetCDF（修正distance维度匹配逻辑）
        """
        # 收集维度信息（包含完整distance）
        meta = self.collect_metadata(radar_objects_list, elevation_configs, drange)
        times, elevs, azis, dists, products = (
            meta["times"], meta["elevations"], meta["azimuths"], meta["distance"], meta["products"]
        )

        # 创建坐标映射字典（用于快速匹配全局索引）
        elev_to_idx = {elev: idx for idx, elev in enumerate(elevs)}
        az_to_idx = {az: idx for idx, az in enumerate(azis)}
        dist_to_idx = {dist: idx for idx, dist in enumerate(dists)}  # 新增：distance值→全局索引映射

        # 创建 NetCDF 文件
        time_dim_attrs = ["scan_type", "pulse_width", "scan_start_time", "cut_number", "nyquist_vel",
                          "pol_type", "products"]
        begin_time = time.time()
        total_time_points = len(radar_objects_list)
        with (Dataset(out_file, "w", format="NETCDF4") as nc):
            # 创建维度
            nc.createDimension("time", len(times))
            nc.createDimension("elevation", len(elevs))
            nc.createDimension("azimuth", len(azis))
            nc.createDimension("distance", len(dists))



            # 写入坐标变量
            tvar = nc.createVariable("time", "f8", ("time",))
            tvar.units = "seconds since 1970-01-01 00:00:00"
            tvar[:] = times.astype(np.int64) // 10 ** 9  # 转换为Unix时间戳（秒）

            nc.createVariable("elevation", "f4", ("elevation",))[:] = elevs
            nc.createVariable("azimuth", "f4", ("azimuth",))[:] = azis
            nc.createVariable("distance", "f4", ("distance",))[:] = dists

            # 定义产品变量（根据数据维度设置分块）
            vars_dict = {}
            # 用第一个数据集的结构作为参考
            sample_ds = radar_objects_list[0].get_multi_elevation_data(elevation_configs[0], drange)
            if sample_ds is None:
                raise ValueError("第一个雷达对象无有效数据，无法定义变量结构")
            original_attrs = sample_ds.attrs  # 提取参考数据集的所有属性
            for key, value in original_attrs.items():
                if key.lower() not in time_dim_attrs:  # 跳过已转为时间维度的属性（不区分大小写）
                    nc.setncattr(key, value)

            for attr in time_dim_attrs:
                # pol_type为整数类型，其他属性根据实际类型调整（如pulse_width可能为float）
                dtype = np.int32 if attr in ["pol_type", "scan_type", "cut_number"] else np.float32
                fill_value = -9999 if dtype == np.int32 else np.nan  # 整数用-9999填充

                vars_dict[attr] = nc.createVariable(
                    attr, dtype, ("time",),  # 维度为(time,)
                    zlib=True, complevel=1,
                    fill_value=fill_value,
                    chunksizes=(1,)  # 按时间分块，提高读写效率
                )
            # 关键步骤：写入属性
            # for key, value in original_attrs.items():
            #     nc.setncattr(key, value)  # 逐个设置属性

            for prod in products:
                if prod not in sample_ds.data_vars:
                    print("没有",prod,sample_ds.data_vars)
                    continue
                data = sample_ds[prod].values
                # 根据数据维度定义变量维度

                if data.ndim == 1:  # (elevation,)
                    vars_dict[prod] = nc.createVariable(
                        prod, "f4", ("time", "elevation"), zlib=True, complevel=1, shuffle=True,
                        chunksizes=(1, len(elevs)), fill_value=np.nan
                    )
                elif data.ndim == 2:  # (elevation, azimuth)
                    vars_dict[prod] = nc.createVariable(
                        prod, "i4", ("time", "elevation", "azimuth"),
                        zlib=True, complevel=1, shuffle=True,
                        chunksizes=(1, 1, len(azis)), fill_value=-9999
                    )
                elif data.ndim == 3:  # (elevation, azimuth, distance)
                    vars_dict[prod] = nc.createVariable(
                        prod, "f4", ("time", "elevation", "azimuth", "distance"), zlib=True,
                        chunksizes=(1, 1, 1024, 230), fill_value=np.nan, complevel=1, shuffle=True
                    )

            # 分块写入数据（逐时间点）
            # for t_idx, (radar_obj, elev_config) in enumerate(zip(radar_objects_list, elevation_configs)):
            for t_idx, (radar_obj, elev_config) in enumerate(tqdm(zip(radar_objects_list, elevation_configs),
                total=total_time_points,
                desc="写入数据",
                unit="时间点"
            )):
                # print(f"写入第 {t_idx + 1}/{len(radar_objects_list)} 个时间点")
                # if t_idx > 10: break
                start_time = time.time()
                ds_start = time.time()
                ds = radar_obj.get_multi_elevation_data(elev_config, drange)
                # if t_idx %10 == 0:
                #     print(ds)
                ds_end = time.time()
                print(f"  获取数据集耗时: {ds_end - ds_start:.4f}s")

                if ds is None:
                    print('跳过无数据的时间点')
                    continue  # 跳过无数据的时间点

                # 处理时间维度属性
                vars_dict["pol_type"][t_idx] = ds.attrs.get("polar_type", -9999)
                vars_dict["scan_type"][t_idx] = ds.attrs.get("scan_type", -9999)
                vars_dict["pulse_width"][t_idx] = ds.attrs.get("pulse_width", -9999)
                vars_dict["scan_start_time"][t_idx] = ds.attrs.get("pulse_width", -9999)
                vars_dict["cut_number"][t_idx] = ds.attrs.get("pulse_width", -9999)

                # 获取当前数据集的坐标（用于匹配全局索引）
                # coords_start = time.time()
                current_elevs = ds.elevation.values
                current_azis = ds.azimuth.values
                current_dists = ds.distance.values  # 当前数据集的distance值

                # 预先计算本次 time 的映射（一次性）
                # 统一四舍五入到6位：与 collect_metadata 中一致
                az_rounded = current_azis
                elev_rounded = current_elevs
                dist_rounded = current_dists

                # boolean masks and local indices
                valid_az_mask = np.isin(az_rounded, list(az_to_idx.keys()))
                valid_dist_mask = np.isin(dist_rounded, list(dist_to_idx.keys()))
                valid_elev_mask = np.isin(elev_rounded, list(elev_to_idx.keys()))


                if not np.any(valid_elev_mask):
                    print(ds.elevation.values)
                    print(list(elev_to_idx.keys()))
                    print("  本时间点无有效仰角，跳过")
                    continue

                if not np.any(valid_az_mask):
                    print(ds.azimuth.values[100:120],max(ds.azimuth.values))
                    print(list(az_to_idx.keys())[100:120],max(list(az_to_idx.keys())))
                    print("  本时间点无有效方角，跳过")
                    continue

                # 当前数据中有效的索引
                valid_az_indices = np.where(valid_az_mask)[0]
                valid_dist_indices = np.where(valid_dist_mask)[0]
                valid_elev_indices = np.where(valid_elev_mask)[0]

                # current_dist_rounded = np.round(current_dists, 6)
                # 全局索引（按顺序对应 local indices）
                valid_az_globals = [az_to_idx[a] for a in az_rounded[valid_az_indices]]
                valid_dist_globals = [dist_to_idx[d] for d in dist_rounded[valid_dist_indices]]
                valid_elev_globals = [elev_to_idx[e] for e in elev_rounded[valid_elev_indices]]

                # coords_end = time.time()
                # print(f"  坐标预处理耗时: {coords_end - coords_start:.4f}s")

                for prod in products:
                    if prod not in ds.data_vars:
                        print('跳过数据集中不存在的产品')
                        continue  # 跳过数据集中不存在的产品
                    data = ds[prod].values
                    # 处理1D数据（elevation,）
                    if data.ndim == 1:
                        # 维度：(elevation,) → 写入 (time, elevation)
                        for e_idx, elev in enumerate(current_elevs):
                            e_global = np.where(np.isclose(elevs, elev, atol=1e-3))[0]
                            if e_global is not None:
                                vars_dict[prod][t_idx, e_global] = data[e_idx]
                            else:
                                print('没有对应位置')

                    # 处理2D数据（仰角, 方位角）
                    elif data.ndim == 2:
                        # # elevation → global
                        # valid_elev_mask = np.isin(current_elevs, list(elev_to_idx.keys()))
                        # valid_elev_indices = np.where(valid_elev_mask)[0]
                        # valid_elev_globals = [elev_to_idx[e] for e in current_elevs[valid_elev_indices]]
                        #
                        # # azimuth → global
                        # valid_az_mask = np.isin(current_azis, list(az_to_idx.keys()))
                        # valid_az_indices = np.where(valid_az_mask)[0]
                        # valid_az_globals = [az_to_idx[az] for az in current_azis[valid_az_indices]]

                        # 写入
                        for e_local, e_global in enumerate(valid_elev_globals):
                            vars_dict[prod][t_idx, e_global, valid_az_globals] = data[
                                valid_elev_indices[e_local], valid_az_indices]

                    # 处理3D数据（elevation, azimuth, distance）
                    elif data.ndim == 3:
                        if len(valid_elev_indices) == 0 or len(valid_az_indices) == 0 or len(valid_dist_indices) == 0:
                            print('没找到数据')
                            continue
                        tmp = data[np.ix_(valid_elev_indices, valid_az_indices, valid_dist_indices)]
                        for i_local, e_global in enumerate(valid_elev_globals):
                            vars_dict[prod][t_idx, e_global, valid_az_globals, valid_dist_globals] = tmp[i_local]
                        # # 遍历当前数据集的有效仰角
                        # elev_rounded = np.round(current_elevs, 6)
                        # valid_elev_mask = np.isin(elev_rounded, list(elev_to_idx.keys()))
                        # valid_elev_indices = np.where(valid_elev_mask)[0]
                        # valid_elev_globals = [elev_to_idx[elev] for elev in elev_rounded[valid_elev_mask]]
                        #
                        # # 匹配方位角
                        # az_rounded = np.round(current_azis, 6)
                        # valid_az_mask = np.isin(az_rounded, list(az_to_idx.keys()))
                        # valid_az_indices = np.where(valid_az_mask)[0]
                        # valid_az_globals = [az_to_idx[az] for az in az_rounded[valid_az_mask]]
                        #
                        # # 写入数据：按(elev, az, dist)的有效全局索引填充
                        # for e_data_idx, e_global in zip(valid_elev_indices, valid_elev_globals):
                        #     # 提取当前仰角下的方位角-距离数据
                        #     az_dist_data = data[e_data_idx, :, :]
                        #     # 只取有效方位角和距离的数据
                        #     valid_data = az_dist_data[valid_az_indices[:, np.newaxis], valid_dist_indices]
                        #     # 写入全局对应位置
                        #     vars_dict[prod][t_idx, e_global, valid_az_globals, valid_dist_globals] = valid_data
                total_time = time.time() - start_time
                print(f"  单时间点耗时: {total_time:.4f}s\n")
        all_time = time.time() - begin_time
        print(f"数据写入完成，文件：{out_file}\n"
              f"总耗时:{all_time:.4f}s")


    def get_multi_time_volume_data(self,radar_objects_list: List,
                                   elevation_configs: List[Dict[int, List[str]]],
                                   drange: float) -> xr.Dataset:
        """
        获取多个时间点的体扫数据并合并（使用并集方式）

        Args:
            radar_objects_list (List): 雷达对象列表，每个对象对应一个时间点
            elevation_configs (List[Dict[int, List[str]]]): 每个时间点的仰角配置列表
            drange (float): 数据半径

        Returns:
            xarray.Dataset: 包含所有时间点、仰角和产品的4D数据集
        """

        if len(radar_objects_list) != len(elevation_configs):
            raise ValueError("雷达对象数量必须与仰角配置数量相等")

        # 收集所有时间点的数据
        time_datasets = []
        time_coords = []

        all_products = set()
        all_elevations = set()
        all_azimuths = set()

        # 第一步：收集每个时间点的数据并建立所有维度的并集
        print("正在处理时间点数据...")
        for i, (radar_obj, elev_config) in enumerate(zip(radar_objects_list, elevation_configs)):
            print(f"处理第 {i + 1}/{len(radar_objects_list)} 个时间点")

            # 获取该时间点的多仰角数据
            time_ds = radar_obj.get_multi_elevation_data(elev_config, drange)
            if time_ds is not None:
                time_datasets.append(time_ds)

                # 解析扫描时间
                scan_time_str = time_ds.attrs.get('scan_time', '')
                try:
                    scan_time = pd.to_datetime(scan_time_str)
                except:
                    scan_time = pd.to_datetime('1900-01-01')
                time_coords.append(scan_time)

                # 收集所有产品类型（排除坐标和辅助变量）
                for product in time_ds.data_vars:
                    if product not in ['longitude', 'latitude', 'height', 'radial_state',
                                       'seq_num', 'rad_num', 'Ele_Num', 'Sec', 'Mic_Sec', 'Moment_um']:
                        all_products.add(product)

                # 收集所有仰角和方位角
                all_elevations.update(time_ds.elevation.values)
                all_azimuths.update(time_ds.azimuth.values)

        if not time_datasets:
            return None

        print(f"{len(time_datasets)} 个时间点的数据")
        print(f"径向: {sorted(all_products)}")
        print(f"仰角数量: {len(all_elevations)}")
        print(f"方位角数量: {len(all_azimuths)}")

        # 创建坐标系统（并集）
        unified_time = pd.DatetimeIndex(time_coords)
        unified_elevation = np.array(sorted(all_elevations))
        unified_azimuth = np.array(sorted(all_azimuths))
        print("方位角:",unified_azimuth)
        # 使用第一个数据集的距离坐标
        distance_coord = time_datasets[0].distance.values

        # 创建4D数据变量 (time, elevation, azimuth, distance)
        data_vars = {}
        all_products = sorted(list(all_products))

        # print("创建4D数据数组...")
        for product in all_products:
            print(f"处理径向: {product}")

            product_data = np.full((len(unified_time), len(unified_elevation),
                                    len(unified_azimuth), len(distance_coord)),
                                   np.nan, dtype=np.float32)

            for time_idx, time_ds in enumerate(time_datasets):
                if product in time_ds.data_vars:
                    # 获取该时间点的数据
                    time_elevations = time_ds.elevation.values
                    time_azimuths = time_ds.azimuth.values
                    time_product_data = time_ds[product].values

                    # 使用numpy的searchsorted或where进行快速映射
                    for elev_idx_local, elev_val in enumerate(time_elevations):
                        elev_idx_global = np.where(np.abs(unified_elevation - elev_val) < 0.01)[0]

                        if len(elev_idx_global) > 0:
                            for az_idx_local, az_val in enumerate(time_azimuths):
                                az_idx_global = np.where(np.abs(unified_azimuth - az_val) < 0.01)[0]

                                if len(az_idx_global) > 0:
                                    if (elev_idx_local < time_product_data.shape[0] and
                                            az_idx_local < time_product_data.shape[1]):
                                        dist_len = min(len(distance_coord), time_product_data.shape[2])
                                        product_data[time_idx, elev_idx_global[0],
                                        az_idx_global[0], :dist_len] = \
                                            time_product_data[elev_idx_local, az_idx_local, :dist_len]

            data_vars[product] = (["time", "elevation", "azimuth", "distance"], product_data)

        # 创建坐标数据（4D）
        # longitude_4d = np.full((len(unified_time), len(unified_elevation),
        #                         len(unified_azimuth), len(distance_coord)), np.nan, dtype=np.float32)
        # latitude_4d = np.full((len(unified_time), len(unified_elevation),
        #                        len(unified_azimuth), len(distance_coord)), np.nan, dtype=np.float32)
        # height_4d = np.full((len(unified_time), len(unified_elevation),
        #                      len(unified_azimuth), len(distance_coord)), np.nan, dtype=np.float32)

        # for time_idx, time_ds in enumerate(time_datasets):
        #     if 'longitude' in time_ds.data_vars:
        #         time_elevations = time_ds.elevation.values
        #         time_azimuths = time_ds.azimuth.values
        #
        #         lon_data = time_ds.longitude.values
        #         lat_data = time_ds.latitude.values
        #         height_data = time_ds.height.values
        #
        #         for elev_idx_local, elev_val in enumerate(time_elevations):
        #             elev_idx_global = np.where(np.abs(unified_elevation - elev_val) < 0.01)[0]
        #
        #             if len(elev_idx_global) > 0:
        #                 for az_idx_local, az_val in enumerate(time_azimuths):
        #                     az_idx_global = np.where(np.abs(unified_azimuth - az_val) < 0.01)[0]
        #
        #                     if len(az_idx_global) > 0:
        #                         if (elev_idx_local < lon_data.shape[0] and
        #                                 az_idx_local < lon_data.shape[1]):
        #                             dist_len = min(len(distance_coord), lon_data.shape[2])
        #                             longitude_4d[time_idx, elev_idx_global[0],
        #                             az_idx_global[0], :dist_len] = lon_data[elev_idx_local, az_idx_local, :dist_len]
        #                             latitude_4d[time_idx, elev_idx_global[0],
        #                             az_idx_global[0], :dist_len] = lat_data[elev_idx_local, az_idx_local, :dist_len]
        #                             height_4d[time_idx, elev_idx_global[0],
        #                             az_idx_global[0], :dist_len] = height_data[elev_idx_local, az_idx_local, :dist_len]

        # 合并所有数据变量
        scan_config_dict = self.scan_config_dict
        all_data_vars = {
            **data_vars,
            "elevation_angles":(["elevation"],scan_config_dict['elevation_angles']),
            # "PRF_1":(["elevation"],scan_config_dict['PRF_1']),
            # "PRF_2":(["elevation"],scan_config_dict['PRF_2']),
            # "Deal_mode":(["elevation"],scan_config_dict['Deal_mode']),
            # "AZI":(["elevation"],scan_config_dict['AZI']),
            # "ELE":(["elevation"],scan_config_dict['ELE']),
            # "St_Angle":(["elevation"],scan_config_dict['St_Angle']),
            # "End_Angle":(["elevation"],scan_config_dict['End_Angle']),
            # "Ang_Res":(["elevation"],scan_config_dict['Ang_Res']),
            # "Scan_Spd":(["elevation"],scan_config_dict['Scan_Spd']),
            # "Log_Res":(["elevation"],scan_config_dict['Log_Res']),
            # "Dop_Res":(["elevation"],scan_config_dict['Dop_Res']),
            # "Max_Range1":(["elevation"],scan_config_dict['Max_Range1']),
            # "Max_Range2":(["elevation"],scan_config_dict['Max_Range2']),
            # "St_Range":(["elevation"],scan_config_dict['St_Range']),
            # "Sample1":(["elevation"],scan_config_dict['Sample1']),
            # "Sample2":(["elevation"],scan_config_dict['Sample2']),
            # "At_Loss":(["elevation"],scan_config_dict['At_Loss']),
            # "Nyq_Spd":(["elevation"],scan_config_dict['Nyq_Spd']),
            # "Dir":(["elevation"],scan_config_dict['Dir']),
            # "Cla_type":(["elevation"],scan_config_dict['Cla_type']),
            # "Fil_type":(["elevation"],scan_config_dict['Fil_type']),
            # "Fil_not_width":(["elevation"],scan_config_dict['Fil_not_width']),
            # "Fil_win":(["elevation"],scan_config_dict['Fil_win']),
            # "longitude": (["time", "elevation", "azimuth", "distance"], longitude_4d),
            # "latitude": (["time", "elevation", "azimuth", "distance"], latitude_4d),
            # "height": (["time", "elevation", "azimuth", "distance"], height_4d)
        }

        # 创建综合属性
        first_ds = time_datasets[0]
        combined_attrs = dict(first_ds.attrs)
        combined_attrs.update({
            "time_range": f"{unified_time[0]} to {unified_time[-1]}",
            "time_count": len(unified_time),
            "elevation_angles": list(unified_elevation),
            "elevation_count": len(unified_elevation),
            "azimuth_count": len(unified_azimuth),
            "products": list(all_products),
        })

        # 创建坐标
        coords = {
            "time": unified_time,
            "elevation": unified_elevation,
            "azimuth": unified_azimuth,
            "distance": distance_coord
        }

        combined_ds = xr.Dataset(all_data_vars, coords=coords, attrs=combined_attrs)

        print(f"数据集维度: {dict(combined_ds.sizes)}")
        return combined_ds

    def collect_metadata(self, radar_objects_list, elevation_configs, drange):
        """
        快速收集维度信息：通过记录最大distance值生成统一坐标
        """
        time_coords = []
        all_elevations = set()
        all_azimuths = set()
        all_products = set()
        max_distance_value = 0.0  # 记录所有数据集中的最大距离值
        first_distance = None  # 记录第一个数据集的完整distance数组（用于步长）
        # auxiliary_fields = {"radial_state", "seq_num", "rad_num",
        #                     "Ele_Num", "Sec", "Mic_Sec", "Moment_um"}

        print("获取维度信息（最大距离优化模式）")
        for i, (radar_obj, elev_config) in enumerate(zip(radar_objects_list, elevation_configs)):
            print(f"读取第{i + 1}/{len(radar_objects_list)}个文件")
            ds = radar_obj.get_multi_elevation_data(elev_config, drange)
            # if i > 10: break
            if ds is None:
                continue

            # 处理时间坐标
            scan_time_str = ds.attrs.get("scan_time", "")
            try:
                scan_time = pd.to_datetime(scan_time_str)
            except Exception:
                scan_time = pd.to_datetime("1900-01-01")
            time_coords.append(scan_time)

            # 收集仰角和方位角
            all_elevations.update(np.round(ds.elevation.values,6))
            all_azimuths.update(np.round(ds.azimuth.values,6))

            # 记录第一个数据集的完整distance（用于获取步长和起点）
            if first_distance is None:
                first_distance = ds.distance.values
                current_max = first_distance[-1] if len(first_distance) > 0 else 0.0
                max_distance_value = current_max

            # 仅更新最大距离值（无需处理整个数组）
            else:
                current_dist = ds.distance.values
                if len(current_dist) > 0:
                    current_max = current_dist[-1]
                    if current_max > max_distance_value:
                        max_distance_value = current_max
                        print(f"更新最大距离值: {max_distance_value:.2f}")

            # 收集产品类型
            for var in ds.data_vars:
                # if var not in auxiliary_fields:
                all_products.add(var)
        # 处理异常情况
        if first_distance is None:
            raise ValueError("未从任何数据集获取到distance信息")

        # 生成统一的distance坐标（基于第一个数据集的步长和全局最大距离）
        # 1. 获取第一个数据集的步长
        if len(first_distance) < 2:
            # 若第一个数据集distance过短，使用drange和默认步长
            step = drange / 1000  # 假设默认1000个距离门
            unified_distance = np.arange(0, max_distance_value + step, step)
        else:
            step = first_distance[1] - first_distance[0]  # 步长=相邻点差值
            # 2. 生成覆盖到最大距离值的坐标
            unified_distance = np.arange(0, max_distance_value + step, step)
            # 确保终点不超过最大距离值（避免浮点误差导致的溢出）
            if unified_distance[-1] > max_distance_value:
                unified_distance = unified_distance[:-1]

        print(f"生成统一distance坐标: 长度={len(unified_distance)}, 范围=[0, {unified_distance[-1]:.2f}]")

        print(f"收集到维度\n"
              f"时间:{len(time_coords)}\n"
              f"仰角:{len(all_elevations)}\n"
              f"方位角:{len(all_azimuths)}\n"
              f"距离:{len(unified_distance)}")
        return {
            "times": pd.DatetimeIndex(time_coords),
            "elevations": np.array(sorted(all_elevations)),
            "azimuths": np.array(sorted(all_azimuths)),
            "distance": unified_distance,
            "products": sorted(list(all_products))
        }

    def collect_metadata_old(self, radar_objects_list, elevation_configs, drange):
        """
        更高效地收集维度信息：
        - distance取所有产品中数据量最大的那个
        - distance不包含0，从reso开始
        """
        time_coords = []
        all_elevations = set()
        all_azimuths = set()
        all_products = set()
        max_ngates = 0
        chosen_reso = None  # 保存最大ngates对应的分辨率

        print("快速收集维度信息...")
        for i, (radar_obj, elev_config) in enumerate(zip(radar_objects_list, elevation_configs)):
            print(f"读取第{i + 1}/{len(radar_objects_list)}个文件")

            # 时间
            scan_time = getattr(radar_obj, "scantime", None)
            if scan_time is None:
                scan_time = pd.to_datetime("1900-01-01")
            time_coords.append(scan_time)

            # 仰角
            for tilt_idx in elev_config.keys():
                all_elevations.add(radar_obj.scan_config[tilt_idx].elev)

            # 方位角（取该tilt的aux）
            sample_tilt = list(elev_config.keys())[0]
            azimuths = np.deg2rad(radar_obj.aux[sample_tilt]["azimuth"])
            all_azimuths.update(azimuths)

            # 距离：找到最大ngates的那个
            for tilt_idx, products in elev_config.items():
                for prod in products:
                    if prod in ["VEL", "SW", "VELSZ"]:
                        reso = radar_obj.scan_config[tilt_idx].dop_reso / 1000
                    else:
                        reso = radar_obj.scan_config[tilt_idx].log_reso / 1000
                    ngates = int(drange // reso)

                    if ngates > max_ngates:
                        max_ngates = ngates
                        chosen_reso = reso

                    all_products.add(prod)

        # 统一distance坐标（从reso开始，不包含0）
        if chosen_reso is None:
            chosen_reso = drange / 1000  # fallback
        unified_distance = np.linspace(chosen_reso, max_ngates * chosen_reso, max_ngates)

        print(f"收集到维度\n"
              f"时间:{len(time_coords)}\n"
              f"仰角:{len(all_elevations)}\n"
              f"方位角:{len(all_azimuths)}\n"
              f"距离:{len(unified_distance)}")
        return {
            "times": pd.DatetimeIndex(time_coords),
            "elevations": np.array(sorted(all_elevations)),
            "azimuths": np.array(sorted(all_azimuths)),
            "distance": unified_distance,
            "products": sorted(list(all_products))
        }



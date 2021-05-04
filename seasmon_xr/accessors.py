from typing import Union

import numpy as np
import xarray

import seasmon_xr.src

__all__ = [
    "IterativeAggregation",
    "PixelAlgorithms",
    "LabelMaker",
    "WhittakerSmoother"
]

@xarray.register_dataarray_accessor("labeler")
class LabelMaker:
    """Class to extending xarray.Dataarray for 'time'

        Adds the properties labelling the time values as either
        dekads or pentads.
    """
    def __init__(self, xarray_obj):
        if not np.issubdtype(xarray_obj, np.datetime64):
            raise TypeError(
            "'.labeler' accessor only available for "
            "DataArray with datetime64 dtype"
            )

        if not hasattr(xarray_obj, "time"):
            raise ValueError("Data array is missing 'time' accessor!")

        if not "time" in xarray_obj.dims:
            xarray_obj = xarray_obj.expand_dims("time")
        self._obj = xarray_obj

    @property
    def dekad(self):
        """Time values labeled as dekads"""
        return self._obj.time.to_series().apply(
                    func=self._gen_labels,
                    args=("d", 10.5),
                ).values

    @property
    def pentad(self):
        """Time values labeled as pentads"""
        return self._obj.time.to_series().apply(
                    func=self._gen_labels,
                    args=("p", 5.19),
                ).values

    @staticmethod
    def _gen_labels(x, l, c):
        return f"{x.year}{x.month:02}" + \
               f"{l}{int(x.day//c+1)}"


@xarray.register_dataset_accessor("iteragg")
@xarray.register_dataarray_accessor("iteragg")
class IterativeAggregation:
    """Class to aggregate multiple coordinate slices"""
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def sum(self,
            n: int = None,
            dim: str = "time",
            begin: Union[str, int, float] = None,
            end: Union[str, int, float] = None,
            method: str = None):
        """Generate sum-aggregations over dim for periods n"""

        yield from self._iteragg(np.nansum, n, dim, begin, end, method)

    def mean(self,
            n: int = None,
            dim: str = "time",
            begin: Union[str, int, float] = None,
            end: Union[str, int, float] = None,
            method: str = None):
        """Generate mean-aggregations over dim for slices of n"""

        yield from self._iteragg(np.nanmean, n, dim, begin, end, method)

    def _iteragg(self, func, n, dim, begin, end, method):

        if dim not in self._obj.dims:
            raise ValueError("Dimension %s doesn't exist in xarray object!" % dim)

        _index = self._obj[dim].to_index()

        if n is None:
            n = self._obj[dim].size
        assert n != 0, "n must be non-zero"

        if begin is not None:
            try:
                begin_ix = (_index.get_loc(begin, method=method) + 1)
            except KeyError:
                raise ValueError("Value %s for 'begin' not found in index for dim %s" % (begin, dim))
        else:
            begin_ix = self._obj.sizes[dim]

        if end is not None:
            try:
                end_ix = _index.get_loc(end, method=method)
            except KeyError:
                raise ValueError("Value %s for 'end' not found in index for dim %s" % (end, dim))
        else:
            end_ix = 0

        for ii in range(begin_ix, 0, -1):
            jj = ii - n
            if ii <= end_ix:
                break
            if jj >= 0 and (ii-jj) == n:
                region = {dim: slice(jj, ii)}
                _obj = self._obj[region].reduce(func, dim)
                _obj = _obj.assign_attrs({
                    "agg_start": str(_index[jj]),
                    "agg_stop": str(_index[ii-1]),
                    "agg_n": _index[jj:ii].size
                })

                if dim == "time":
                    _obj = _obj.expand_dims(time=[self._obj.time[ii-1].values])
                yield _obj

@xarray.register_dataset_accessor("whit")
@xarray.register_dataarray_accessor("whit")
class WhittakerSmoother:
    """Class for applying different version of the Whittaker smoother to a
       Dataset or DataArray"""
    def __init__(self, xarray_obj):
        if not "time" in xarray_obj.dims:
            raise ValueError(
                "'.whit' can only be applied to datasets / dataarrays "
                "with 'time' dimension!"
            )

        self._obj = xarray_obj


    def whits(self,
              nodata: Union[int, float],
              sg: xarray.DataArray = None,
              s: float = None,
              p: float = None) -> xarray.Dataset:

        """Apply whittaker with fixed S

        Fixed S can be either provided as constant or
        as sgrid with a constant per pixel

        Args:
            ds: input dataset,
            nodata: nodata value
            sg: sgrid,
            s: S value
            p: Envelope value for asymmetric weights

        Returns:
            ds_out: xarray.Dataset with smoothed data"""

        if sg is None and s is None:
            raise ValueError("Need S or sgrid")

        lmda = sg if sg is not None else s

        if p is not None:

            xout = xarray.apply_ufunc(
                seasmon_xr.src.ws2dpgu,
                self._obj,
                lmda,
                nodata,
                p,
                input_core_dims=[["time"], [], [], []],
                output_core_dims=[["time"]],
                dask="parallelized",
                keep_attrs=True,
            )

        else:

            xout = xarray.apply_ufunc(
                seasmon_xr.src.ws2dgu,
                self._obj,
                lmda,
                nodata,
                input_core_dims=[["time"], [], []],
                output_core_dims=[["time"]],
                dask="parallelized",
                keep_attrs=True,
            )

        return xout

    def whitsvc(self,
                nodata: Union[int, float],
                lc: xarray.DataArray = None,
                srange: np.ndarray = None,
                p: float = None) -> xarray.Dataset:
        """Apply whittaker with V-curve optimization of S

        Args:
            dim: dimension to use for filtering
            nodata: nodata value
            lc: lag1 autocorrelation DataArray,
            srange: values of S for V-curve optimization (mandatory if no autocorrelation raster)
            p: Envelope value for asymmetric weights

        Returns:
            ds_out: xarray.Dataset with smoothed data and sgrid
        """


        if lc is not None:
            if p is None:
                raise ValueError("If lc is set, a p value needs to be specified as well.")

            ds_out, sgrid = xarray.apply_ufunc(
                seasmon_xr.src.ws2doptvplc,
                self._obj,
                nodata,
                p,
                lc,
                input_core_dims=[["time"], [], [], []],
                output_core_dims=[["time"], []],
                dask="parallelized",
                keep_attrs=True,
            )

        else:

            if srange is None:
                raise ValueError("Need either lagcorr or srange!")

            if p:
                ds_out, sgrid = xarray.apply_ufunc(
                                    seasmon_xr.src.ws2doptvp,
                                    self._obj,
                                    nodata,
                                    p,
                                    srange,
                                    input_core_dims=[["time"], [], [], ["dim0"]],
                                    output_core_dims=[["time"], []],
                                    dask="parallelized",
                                    keep_attrs=True,
                )

            else:

                ds_out, sgrid = xarray.apply_ufunc(
                                    seasmon_xr.src.ws2doptv,
                                    self._obj,
                                    nodata,
                                    srange,
                                    input_core_dims=[["time"], [], ["dim0"]],
                                    output_core_dims=[["time"], []],
                                    dask="parallelized",
                                    keep_attrs=True,
                            )

        ds_out = ds_out.to_dataset()
        ds_out["sgrid"] = np.log10(sgrid).astype("float32")

        return ds_out

    def whitint(self,
                labels_daily: np.ndarray,
                template: np.ndarray):
        """Wrapper for temporal interpolation using the Whittaker filter"""
        if self._obj.dtype != "int16":
            raise NotImplementedError("Temporal interpolation works currently only with int16 input!")

        template_out = np.zeros(np.unique(labels_daily).size, dtype="u1")

        ds_out = xarray.apply_ufunc(
            seasmon_xr.src.tinterpolate,
            self._obj,
            template,
            labels_daily,
            template_out,
            input_core_dims=[["time"], ["dim0"], ["dim1"], ["dim2"]],
            output_core_dims=[["newtime"]],
            dask_gufunc_kwargs={"output_sizes": {"newtime": template_out.size}},
            output_dtypes=["int16"],
            dask="parallelized",
            keep_attrs=True,
        )

        return ds_out

@xarray.register_dataset_accessor("algo")
@xarray.register_dataarray_accessor("algo")
class PixelAlgorithms:
    """Set of algorithms to be applied to pixel timeseries"""
    def __init__(self, xarray_obj):
        if not "time" in xarray_obj.dims:
            raise ValueError(
                "'.algo' can only be applied to datasets / dataarrays "
                "with 'time' dimension!"
            )

        if xarray_obj.dims[0] != "time":
            xarray_obj = xarray_obj.transpose("time", ...)
        self._obj = xarray_obj

    def spi(self):
        """Calculates the SPI along the time dimension"""
        return xarray.apply_ufunc(
                    seasmon_xr.src.spifun,
                    self._obj,
                    dask="parallelized"
                )

    def lroo(self):
        """Longest run of ones along time dimension"""

        return xarray.apply_ufunc(
                    seasmon_xr.src.lroo,
                    self._obj,
                    input_core_dims=[["time"]],
                    dask="parallelized",
                    output_dtypes=['uint8'],
                )

    def autocorr(self):
        """Calculates the autocorrelation along time

        Returns:
            xarray.DataArray with lag1 autocorrelation
        """

        return xarray.apply_ufunc(
                    seasmon_xr.src.autocorr,
                    self._obj,
                    input_core_dims=[["time"]],
                    dask="parallelized",
                    output_dtypes=["float32"]
                )

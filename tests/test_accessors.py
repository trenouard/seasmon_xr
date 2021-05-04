"""Tests for xarray accessors"""
#pylint: disable=redefined-outer-name,unused-import,no-member,no-name-in-module
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import seasmon_xr
from seasmon_xr.src.ws2d import ws2d

@pytest.fixture
def darr():
    np.random.seed(42)
    x = xr.DataArray(
        np.random.randint(1,100, (5 ,2 ,2)),
        dims=("time", "y", "x"),
        name='band'
    )
    x['time'] = pd.date_range(start="2000-01-01", periods=5, freq="10D")

    return x

@pytest.fixture
def res_spi():
    return np.array([[[  217,  1401],
                    [-1673,   988]],
                    [[  397,  -979],
                    [  943,  1178]],
                    [[  644,   979],
                    [ 1061,   103]],
                    [[-1947,  -920],
                    [  116, -1139]],
                    [[  847,  -508],
                    [ -424, -1139]]])

def test_labels_dekad(darr):

    np.testing.assert_array_equal(
        darr.time.labeler.dekads,
        ['200001d1', '200001d2', '200001d3', '200001d3', '200002d1']
    )

def test_labels_dekad_single(darr):

    np.testing.assert_array_equal(
        darr.isel(time=0).time.labeler.dekads,
        ['200001d1']
    )

def test_labels_pentad(darr):

    np.testing.assert_array_equal(
        darr.time.labeler.pentads,
        ['200001p1', '200001p3', '200001p5', '200001p6', '200002p2']
    )

def test_labels_pentad_single(darr):

    np.testing.assert_array_equal(
        darr.isel(time=0).time.labeler.pentads,
        ['200001p1']
    )

def test_algo_lroo(darr):
    _res = np.array([[3, 0],
                    [4, 2]],
                    dtype="uint8"
    )

    darr_lroo = ((darr>30)*1).astype("uint8").algo.lroo()
    assert isinstance(darr_lroo, xr.DataArray)
    np.testing.assert_array_equal(darr_lroo, _res)


def test_algo_autocorr(darr):
    _res = np.array([[-0.7395127 , -0.69477093],
                     [-0.1768683 ,  0.6412078 ]],
                    dtype="float32"
    )
    darr_autocorr = darr.algo.autocorr()
    assert isinstance(darr_autocorr, xr.DataArray)
    np.testing.assert_almost_equal(darr_autocorr, _res)

def test_algo_spi(darr, res_spi):
    _res = darr.algo.spi()
    assert isinstance(_res, xr.DataArray)
    np.testing.assert_array_equal(_res, res_spi)

def test_algo_spi_transp(darr, res_spi):
    _darr = darr.transpose(..., "time")
    _res = _darr.algo.spi()
    assert isinstance(_res, xr.DataArray)
    np.testing.assert_array_equal(_res, res_spi)

def test_agg_sum_1(darr):
    n = 0
    ii = -1
    for _x in darr.iteragg.sum(1):
        np.testing.assert_array_equal(
            _x.squeeze(),
            darr[ii, :]
        )
        assert _x.attrs['agg_n'] == 1
        assert _x.time == darr.time[ii]
        ii -=1
        n += 1

def test_agg_sum_1_lim_begin(darr):
    n = 0
    begin = "2000-01-21"
    ii = (darr.time.to_index().get_loc(begin))
    for _x in darr.iteragg.sum(1, begin=begin):
        np.testing.assert_array_equal(
            _x.squeeze(),
            darr[ii, :]
        )
        assert _x.attrs['agg_n'] == 1
        assert _x.time == darr.time[ii]
        ii -=1
        n += 1
    assert n == 3

def test_agg_sum_1_lim_end(darr):
    n = 0
    end = "2000-01-21"
    ii = -1
    for _x in darr.iteragg.sum(1, end=end):
        np.testing.assert_array_equal(
            _x.squeeze(),
            darr[ii, :]
        )
        assert _x.attrs['agg_n'] == 1
        assert _x.time == darr.time[ii]
        ii -=1
        n += 1
    assert n == 3
    assert _x.time.dt.strftime("%Y-%m-%d").values == end #pylint: disable=undefined-loop-variable

def test_agg_sum_1_lim_begin_end(darr):
    n = 0
    begin = "2000-01-31"
    end = "2000-01-11"
    ii = (darr.time.to_index().get_loc(begin))
    for _x in darr.iteragg.sum(1, begin=begin, end=end):
        np.testing.assert_array_equal(
            _x.squeeze(),
            darr[ii, :]
        )
        assert _x.attrs['agg_n'] == 1
        assert _x.time == darr.time[ii]
        ii -=1
        n += 1
    assert n == 3
    assert _x.time.dt.strftime("%Y-%m-%d").values == end #pylint: disable=undefined-loop-variable

def test_agg_sum_3(darr):
    n = 0
    for _x in darr.iteragg.sum(n=3):
        np.testing.assert_array_equal(
            _x.squeeze(),
            darr.sel(time=slice(_x.attrs['agg_start'], _x.attrs['agg_stop'])).sum("time")
        )
        assert _x.attrs['agg_n'] == 3
        assert str(_x.time.to_index()[0]) == _x.attrs['agg_stop']
        n += 1
    assert n == 3

def test_agg_mean_1(darr):
    n = 0
    ii = -1
    for _x in darr.iteragg.mean(1):
        np.testing.assert_array_equal(
            _x.squeeze(),
            darr[ii, :]
        )
        assert _x.attrs['agg_n'] == 1
        assert _x.time == darr.time[ii]
        ii -=1
        n += 1
    assert n == 5

def test_agg_mean_1_lim_begin(darr):
    n = 0
    begin = "2000-01-21"
    ii = (darr.time.to_index().get_loc(begin))
    for _x in darr.iteragg.mean(1, begin=begin):
        np.testing.assert_array_equal(
            _x.squeeze(),
            darr[ii, :]
        )
        assert _x.attrs['agg_n'] == 1
        assert _x.time == darr.time[ii]
        ii -=1
        n += 1
    assert n == 3

def test_agg_mean_1_lim_end(darr):
    n = 0
    end = "2000-01-21"
    ii = -1
    for _x in darr.iteragg.mean(1, end=end):
        np.testing.assert_array_equal(
            _x.squeeze(),
            darr[ii, :]
        )
        assert _x.attrs['agg_n'] == 1
        assert _x.time == darr.time[ii]
        ii -=1
        n += 1
    assert n == 3
    assert _x.time.dt.strftime("%Y-%m-%d").values == end #pylint: disable=undefined-loop-variable

def test_agg_mean_1_lim_begin_end(darr):
    n = 0
    begin = "2000-01-31"
    end = "2000-01-11"
    ii = (darr.time.to_index().get_loc(begin))
    for _x in darr.iteragg.mean(1, begin=begin, end=end):
        np.testing.assert_array_equal(
            _x.squeeze(),
            darr[ii, :]
        )
        assert _x.attrs['agg_n'] == 1
        assert _x.time == darr.time[ii]
        ii -=1
        n += 1
    assert n == 3
    assert _x.time.dt.strftime("%Y-%m-%d").values == end #pylint: disable=undefined-loop-variable

def test_agg_mean_3(darr):
    n = 0
    for _x in darr.iteragg.mean(3):
        np.testing.assert_array_equal(
            _x.squeeze(),
            darr.sel(time=slice(_x.attrs['agg_start'], _x.attrs['agg_stop'])).mean("time")
        )
        assert _x.attrs['agg_n'] == 3
        assert str(_x.time.to_index()[0]) == _x.attrs['agg_stop']
        n += 1
    assert n == 3

def test_whit_whits_s(darr):
    _res = np.array([[[54, 54, 55, 56, 60],
                    [74, 60, 48, 35, 24]],
                    [[49, 56, 59, 58, 56],
                    [82, 60, 37, 14, -7]]],
                    dtype="int16",
    )
    _darr = darr.whit.whits(nodata=0, s=10)
    np.testing.assert_array_equal(_darr, _res)

    y = darr[:,0,0].astype("float64").data
    w = ((y!=0)*1).astype("float64")
    z = np.rint(seasmon_xr.src.ws2d.ws2d(y,10.0, w))

    np.testing.assert_array_equal(_darr[0,0,:].data, z)

def test_whit_whits_sg(darr):
    sg = np.full((2,2), -0.5)
    _res = np.array([[[76, 32, 35, 75, 61],
                    [78, 69, 31, 27, 36]],
                    [[8, 65, 109, 91, 4],
                    [ 93, 43, 36, 25, -10]]],
                    dtype="int16",
    )
    _darr = darr.whit.whits(nodata=0, sg=sg)
    np.testing.assert_array_equal(_darr, _res)

    y = darr[:,0,0].astype("float64").data
    w = ((y!=0)*1).astype("float64")
    l = -0.5
    z = np.rint(seasmon_xr.src.ws2d.ws2d(y,l,w))

    np.testing.assert_array_equal(_darr[0,0,:].data, z)

def test_whit_whitsvc(darr):
    srange = np.arange(-2, 2)
    _res = np.array([[[55, 59, 55, 38, 72],
                    [82, 49, 50, 34, 27]],
                    [[26, 71, 81, 61, 39],
                    [80, 69, 33, 8, -3]]],
                    dtype="int16",
    )
    _darr = darr.whit.whitsvc(nodata=0, srange=srange)
    assert isinstance(_darr, xr.Dataset)
    assert "band" in _darr
    assert "sgrid" in _darr

    np.testing.assert_array_equal(_darr.band, _res)
    np.testing.assert_array_equal(_darr.sgrid, np.full((2,2), -0.5))

    y = darr[:,0,0].astype("float64").data

    z, l = seasmon_xr.src.ws2doptv(y,0.0, srange)

    np.testing.assert_array_equal(_darr.band[0,0,:].data, np.rint(z))
    assert np.log10(l) == -0.5

def test_whit_whitsvcp(darr):
    srange = np.arange(-2, 2)
    _res = np.array([[[54, 66, 71, 49, 86],
                    [92, 60, 70, 43, 30]],
                    [[60, 79, 83, 72, 55],
                    [85, 84, 40, 10,  1]]],
                    dtype="int16",
    )
    _darr = darr.whit.whitsvc(nodata=0, srange=srange, p=0.90)
    assert isinstance(_darr, xr.Dataset)
    assert "band" in _darr
    assert "sgrid" in _darr

    np.testing.assert_array_equal(_darr.band, _res)
    np.testing.assert_array_equal(_darr.sgrid, np.array([[-1.5, -1.5],[-0.5, -1.5]]))

    y = darr[:,0,0].astype("float64").data

    z, l = seasmon_xr.src.ws2doptvp(y, 0.0, 0.90, srange)

    np.testing.assert_array_equal(_darr.band[0,0,:].data, np.rint(z))
    assert np.log10(l) == -1.5


def test_whit_whitint(darr):
    labels_daily = np.array([
            2000011, 2000011, 2000011, 2000011, 2000011, 2000011, 2000011,
            2000011, 2000011, 2000011, 2000012, 2000012, 2000012, 2000012,
            2000012, 2000012, 2000012, 2000012, 2000012, 2000012, 2000013,
            2000013, 2000013, 2000013, 2000013, 2000013, 2000013, 2000013,
            2000013, 2000013, 2000013, 2000021, 2000021, 2000021, 2000021,
            2000021, 2000021, 2000021, 2000021, 2000021, 2000021, 2000022,
            2000022, 2000022, 2000022, 2000022, 2000022, 2000022, 2000022,
            2000022, 2000022],
            dtype="int32"
    )

    template = np.array([
        0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]
    )

    temp = template.copy()
    res = darr.astype("int16").whit.whitint(labels_daily, temp)
    assert "newtime" in res.dims
    assert res.newtime.size == 5

    xx = darr[:,0,0].values

    w = template.copy()
    temp = template.copy()

    for ix, ii in enumerate(np.where(template!=0)[0]):
        template[ii] = xx[ix]
    template[-1] = xx[-1]

    z = ws2d(template, 0.00001, w)

    _df = pd.DataFrame({"data": z, "labels": labels_daily})
    xx_int = _df.groupby("labels").mean().round().astype("int16")

    np.testing.assert_array_equal(
        res[0,0,:],
        xx_int.data.values
    )

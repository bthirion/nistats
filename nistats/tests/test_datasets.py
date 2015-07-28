import os
import shutil
from numpy import asarray
from nose.tools import assert_true, assert_false, assert_equal, assert_raises
from nose import with_setup
from tempfile import mkdtemp

from nilearn._utils.testing import (mock_request, wrap_chunk_read_,
                                    FetchFilesMock, assert_raises_regex)
from nilearn._utils.compat import _basestring
from nilearn.datasets import utils, func

from nistats import datasets


original_fetch_files = None
original_url_request = None
original_chunk_read = None

mock_fetch_files = None
mock_url_request = None
mock_chunk_read = None

currdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(currdir, 'data')
tmpdir = None


def setup_mock():
    global original_url_request
    global mock_url_request
    mock_url_request = mock_request()
    original_url_request = utils._urllib.request
    utils._urllib.request = mock_url_request

    global original_chunk_read
    global mock_chunk_read
    mock_chunk_read = wrap_chunk_read_(utils._chunk_read_)
    original_chunk_read = utils._chunk_read_
    utils._chunk_read_ = mock_chunk_read

    global original_fetch_files
    global mock_fetch_files
    mock_fetch_files = FetchFilesMock()
    original_fetch_files = func._fetch_files
    func._fetch_files = mock_fetch_files


def teardown_mock():
    global original_url_request
    utils._urllib.request = original_url_request

    global original_chunk_read
    utils._chunk_read_ = original_chunk_read

    global original_fetch_files
    func._fetch_files = original_fetch_files


def setup_tmpdata():
    # create temporary dir
    global tmpdir
    tmpdir = mkdtemp()


def teardown_tmpdata():
    # remove temporary dir
    global tmpdir
    if tmpdir is not None:
        shutil.rmtree(tmpdir)


@with_setup(setup_tmpdata, teardown_tmpdata)
@with_setup(setup_mock, teardown_mock)
def test_fetch_localizer():
    local_url = "file://" + datadir
    ids = asarray([('S%2d' % i).encode() for i in range(94)])
    ids = ids.view(dtype=[('subject_id', 'S3')])
    mock_fetch_files.add_csv('cubicwebexport.csv', ids)
    mock_fetch_files.add_csv('cubicwebexport2.csv', ids)

    # Disabled: cannot be tested without actually fetching covariates CSV file
    # All subjects
    dataset = datasets.fetch_localizer_first_level(data_dir=tmpdir)
    assert_true(isinstance(dataset.paradigm, _basestring))
    assert_true(isinstance(dataset.epi_img, _basestring))


#@with_setup(setup_tmpdata, teardown_tmpdata)
#@with_setup(setup_mock, teardown_mock)
def test_fetch_spm_auditory():
    dataset = datasets.fetch_spm_auditory(data_dir=tmpdir)
    assert_true(isinstance(dataset.anat, _basestring))
    assert_true(isinstance(dataset.func[0], _basestring))
    assert_equal(len(dataset.func), 96)


#@with_setup(setup_tmpdata, teardown_tmpdata)
#@with_setup(setup_mock, teardown_mock)
def test_fetch_spm_multimodal():
    dataset = datasets.fetch_spm_multimodal_fmri(data_dir=tmpdir)
    assert_true(isinstance(dataset.anat, _basestring))
    assert_true(isinstance(dataset.func1[0], _basestring))
    assert_equal(len(dataset.func1), 390)
    assert_true(isinstance(dataset.func2[0], _basestring))
    assert_equal(len(dataset.func2), 390)
    assert_equal(dataset.slice_order, 'descending')
    assert_true(dataset.trials_ses1, _basestring)
    assert_true(dataset.trials_ses2, _basestring)

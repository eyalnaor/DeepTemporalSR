from __future__ import division

try:
    import resource
except ImportError:
    pass
import numpy as np
from imresize import imresize
import subprocess as sp
import time
from argparse import ArgumentParser
from PIL import Image
import glob
import os
import shutil
from threading import Thread

try:
    import torch_resizer
    import torch
except ImportError:
    pass

try:
    from queue import Queue, Empty
except ImportError:
    # python 2 compatibility
    from Queue import Queue, Empty


bp_debug = False


def space_time_backprojection(hfr_hr_pred, lfr_hr_in, hfr_lr_in, device):
    # inputs are in H-W-T-C np arrays
    _rusage('init')
    if bp_debug:
        print('-stbp- hfr_hr_pred {}, lfr_hr_in {}, hfr_lr_in {}'
              .format(hfr_hr_pred.shape if hfr_hr_pred is not None else None,
                      lfr_hr_in.shape if lfr_hr_in is not None else None,
                      hfr_lr_in.shape if hfr_lr_in is not None else None))

    num_iters = 1

    if hfr_hr_pred is None:
        # hfr_hr_pred = imresize(hfr_lr_in, scale_factor=[2., 2., 1.], kernel='cubic')
        # hfr_hr_pred = imresize(hfr_lr_in, output_shape=[lfr_hr_in.shape[0], lfr_hr_in.shape[1], hfr_lr_in.shape[2], hfr_lr_in.shape[3]], kernel='cubic')
        resizer = torch_resizer.Resizer(hfr_lr_in.shape, output_shape=[lfr_hr_in.shape[0], lfr_hr_in.shape[1], hfr_lr_in.shape[2], hfr_lr_in.shape[3]],
                                             kernel='cubic', antialiasing=True, device='cuda', dtype=torch.float16)
        hfr_hr_pred = resizer(torch.tensor(hfr_lr_in, dtype=torch.float16).to(device)).cpu().numpy()
    _rusage('make pred')
    # CHECK!!
    _check_inputs(hfr_hr_pred, lfr_hr_in, hfr_lr_in)

    for it in range(num_iters):
        hfr_hr_pred = temporal_backprojection_np(hfr_hr_pred, lfr_hr_in)
        _rusage('{} tbp'.format(it))
        hfr_hr_pred = spatial_backprojection_np(hfr_hr_pred, hfr_lr_in, device)
        _rusage('{} sbp'.format(it))
    # final step is temporal
    hfr_hr_pred = temporal_backprojection_np(hfr_hr_pred, lfr_hr_in)

    _rusage('finally')
    return np.clip(hfr_hr_pred, 0., 1.)


def space_time_backprojection_cpu(hfr_hr_pred, lfr_hr_in, hfr_lr_in):
    # inputs are in H-W-T-C np arrays
    _rusage('init')
    if bp_debug:
        print('-stbp- hfr_hr_pred {}, lfr_hr_in {}, hfr_lr_in {}'
              .format(hfr_hr_pred.shape if hfr_hr_pred is not None else None,
                      lfr_hr_in.shape if lfr_hr_in is not None else None,
                      hfr_lr_in.shape if hfr_lr_in is not None else None))

    num_iters = 1

    if hfr_hr_pred is None:
        # hfr_hr_pred = imresize(hfr_lr_in, scale_factor=[2., 2., 1.], kernel='cubic')
        hfr_hr_pred = imresize(hfr_lr_in, output_shape=[lfr_hr_in.shape[0], lfr_hr_in.shape[1], hfr_lr_in.shape[2], hfr_lr_in.shape[3]], kernel='cubic')
        # resizer = torch_resizer.Resizer(hfr_lr_in.shape, output_shape=[lfr_hr_in.shape[0], lfr_hr_in.shape[1], hfr_lr_in.shape[2], hfr_lr_in.shape[3]],
        #                                      kernel='cubic', antialiasing=True, device='cuda', dtype=torch.float16)
        # hfr_hr_pred = resizer(torch.tensor(hfr_lr_in, dtype=torch.float16).to(device)).cpu().numpy()
    _rusage('make pred')
    # CHECK!!
    _check_inputs(hfr_hr_pred, lfr_hr_in, hfr_lr_in)

    for it in range(num_iters):
        hfr_hr_pred = temporal_backprojection_np(hfr_hr_pred, lfr_hr_in)
        _rusage('{} tbp'.format(it))
        hfr_hr_pred = spatial_backprojection_np_cpu(hfr_hr_pred, hfr_lr_in)
        _rusage('{} sbp'.format(it))
    # final step is temporal
    hfr_hr_pred = temporal_backprojection_np(hfr_hr_pred, lfr_hr_in)

    _rusage('finally')
    return np.clip(hfr_hr_pred, 0., 1.)

def temporal_expand_numpy_hwtc(x, rate):
    return np.repeat(x, rate, axis=2)


def temporal_blur_numpy_hwtc(x, rate):
    # discard (!) frames if duration of x is not divisable by rate
    t = (x.shape[2] // rate) * rate
    return x[..., :t, :].reshape(x.shape[0], x.shape[1], t // rate, rate, x.shape[3]).mean(axis=3)


def temporal_backprojection_np(hfr_pred, lfr_in):
    # hfr_pred, lfr_in are np arrays with dimensions h-w-t-c, assuming only shape difference is in t dimension
    rate = hfr_pred.shape[2] // lfr_in.shape[2]
    assert(lfr_in.shape[2] * rate == hfr_pred.shape[2])  # rate should be integer
    bp = hfr_pred - temporal_expand_numpy_hwtc(temporal_blur_numpy_hwtc(hfr_pred, rate) - lfr_in, rate)

    return bp


def spatial_backprojection_np(hr_pred, lr_in, device):
    lrsz = lr_in[..., 0, :].shape
    hrsz = hr_pred[..., 0, :].shape

    resizer = torch_resizer.Resizer(hr_pred[..., 0, :].shape, output_shape=lrsz, kernel='cubic', antialiasing=True, device='cuda', dtype=torch.float16)
    resizer2 = torch_resizer.Resizer(lrsz, output_shape=hrsz, kernel='cubic', antialiasing=True, device='cuda', dtype=torch.float16)


    for fidx in range(hr_pred.shape[2]):
        # err_ = imresize(hr_pred[..., fidx, :], output_shape=lrsz, kernel='cubic') - lr_in[..., fidx, :]
        # hr_pred[..., fidx, :] -= imresize(err_, output_shape=hrsz, kernel='cubic')
        err_ = resizer(torch.tensor(hr_pred[..., fidx, :], dtype=torch.float16).to(device)).cpu().numpy() - lr_in[..., fidx, :]
        hr_pred[..., fidx, :] -= resizer2(torch.tensor(err_, dtype=torch.float16).to(device)).cpu().numpy()
    return hr_pred

def spatial_backprojection_np_cpu(hr_pred, lr_in):
    lrsz = lr_in[..., 0, :].shape
    hrsz = hr_pred[..., 0, :].shape

    # resizer = torch_resizer.Resizer(hr_pred[..., 0, :].shape, output_shape=lrsz, kernel='cubic', antialiasing=True, device='cuda', dtype=torch.float16)
    # resizer2 = torch_resizer.Resizer(lrsz, output_shape=hrsz, kernel='cubic', antialiasing=True, device='cuda', dtype=torch.float16)


    for fidx in range(hr_pred.shape[2]):
        err_ = imresize(hr_pred[..., fidx, :], output_shape=lrsz, kernel='cubic') - lr_in[..., fidx, :]
        hr_pred[..., fidx, :] -= imresize(err_, output_shape=hrsz, kernel='cubic')
        # err_ = resizer(torch.tensor(hr_pred[..., fidx, :], dtype=torch.float16).to(device)).cpu().numpy() - lr_in[..., fidx, :]
        # hr_pred[..., fidx, :] -= resizer2(torch.tensor(err_, dtype=torch.float16).to(device)).cpu().numpy()
    return hr_pred


def _check_inputs(hfr_hr_pred, lfr_hr_in, hfr_lr_in):
    def _check_hwtc(vol):
        assert(vol.ndim == 4 and (vol.shape[3] == 3 or vol.shape[3] == 1))
    if hfr_hr_pred is not None:
        _check_hwtc(hfr_hr_pred)
    _check_hwtc(lfr_hr_in)
    _check_hwtc(hfr_lr_in)

    # check temporal dimensions
    assert(hfr_lr_in.shape[2] > lfr_hr_in.shape[2])
    rate = hfr_lr_in.shape[2] // lfr_hr_in.shape[2]
    assert(lfr_hr_in.shape[2] * rate == hfr_lr_in.shape[2])
    if hfr_hr_pred is not None:
        assert(hfr_hr_pred.shape[2] == hfr_lr_in.shape[2])

    # check spatial dimensions
    for dim in range(2):
        assert(lfr_hr_in.shape[dim] > hfr_lr_in.shape[dim])
        if hfr_hr_pred is not None:
            assert(hfr_hr_pred.shape[dim] == lfr_hr_in.shape[dim])
    #print('inputs checked okay')


# -------------------------------------------------------------------
# distributed processing
# -------------------------------------------------------------------
def shai_bagons_bad_ass_distributed_space_time_backprojection(hfr_hr_pred, lfr_hr_in, hfr_lr_in, base_folder):
    # inputs are in H-W-T-C np arrays
    _rusage('badbp init')
    t = time.time()
    if bp_debug:
        print('-badbp- hfr_hr_pred {}, lfr_hr_in {}, hfr_lr_in {}'
              .format(hfr_hr_pred.shape if hfr_hr_pred is not None else None,
                      lfr_hr_in.shape if lfr_hr_in is not None else None,
                      hfr_lr_in.shape if hfr_lr_in is not None else None))

    rate = hfr_lr_in.shape[2] // lfr_hr_in.shape[2]
    assert(rate > 1)
    #print('-badbp- submittinig with temporal sample rate = {}'.format(rate))
    max_chunk_size = max(24 // rate, 1)  # number of hfr frames to process at a chunk. at least rate frames
    tag = '{}/{}x{}x{}/'.format(base_folder, lfr_hr_in.shape[0], lfr_hr_in.shape[1], hfr_lr_in.shape[2])

    num_workers = max(8, min(24, lfr_hr_in.shape[2] // max_chunk_size))
    cid = 0
    fr_ = 0

    chunk_q = Queue()
    jid_q = Queue()
    err_q = Queue()
    # global data
    # allocate room for output
    hfr_hr_pred = np.zeros([lfr_hr_in.shape[0], lfr_hr_in.shape[1], hfr_lr_in.shape[2], hfr_lr_in.shape[3]], dtype=np.float32)
    _rusage('badbp allocating out')

    # multi threading submission
    class Worker(Thread):
        def __init__(self, inq, outq, errq):
            super(Worker, self).__init__()
            self.inq = inq
            self.outq = outq
            self.errq = errq
            self.daemon = True
            self.start()

        def run(self):
            # first part - submit all chunks
            while True:
                item = self.inq.get()
                if item is None:
                    # "stop" signal
                    self.inq.task_done()
                    break
                fr_, to_, cid = item
                hfr_lr_folder = '{}-c{}-hfr_lr'.format(tag, cid)
                _write_chunk(hfr_lr_in[..., fr_*rate:to_*rate, :], hfr_lr_folder)
                lfr_hr_folder = '{}-c{}-lfr_hr'.format(tag, cid)
                _write_chunk(lfr_hr_in[..., fr_:to_, :], lfr_hr_folder)
                pyargs = '--lfr_hr_in {} --hfr_lr_in {}'.format(lfr_hr_folder, hfr_lr_folder)
                if hfr_hr_pred is not None:
                    hfr_hr_folder = '{}-c{}-hfr_hr_in'.format(tag, cid)
                    _write_chunk(hfr_hr_pred[..., fr_*rate:to_*rate, :], hfr_hr_folder)
                    pyargs = '{} --hfr_hr_pred {}'.format(pyargs, hfr_hr_folder)
                output_folder = '{}-c{}-output'.format(tag, cid)
                pyargs = '{} --output {}'.format(pyargs, output_folder)
                jid = _submit_job_and_get_jid(pyargs, tag)
                self.outq.put([jid, output_folder, fr_, to_])
                self.inq.task_done()
            # second part - gather
            while True:
                item = self.outq.get()
                if item is None:
                    # "stop" signal
                    self.outq.task_done()
                    break
                jid, output_folder, fr_, to_ = item

                status = _check_job_status(jid)
                if status == 'done':
                    #print('chunk {}:{} done!'.format(fr_, to_))
                    hfr_hr_pred[..., rate*fr_:rate*to_, :] = _read_chunk(output_folder, (to_ - fr_)*rate)
                elif status == 'exit':
                    #print('chunk {}:{} jid={} failed filling in rubish'.format(fr_, to_, jid))
                    hfr_hr_pred[..., rate * fr_:rate * to_, :] = 0.5
                    try:
                        with open('{}{}.e'.format(tag, jid), 'r') as R:
                            for l in R.readlines():
                                z = 0
                                #print('\t-jid{}-{}'.format(jid, l.rstrip()))
                    except:
                        pass
                    self.errq.put(jid)
                else:
                    #print('\t jid {} status {}'.format(jid, status))
                    # resubmit
                    self.outq.put(item)
                self.outq.task_done()
                time.sleep(1)

    # start the workers
    workers = [Worker(chunk_q, jid_q, err_q) for _ in range(num_workers)]
    _rusage('badbp workers started')

    while fr_ < lfr_hr_in.shape[2]:
        to_ = min(lfr_hr_in.shape[2], fr_ + max_chunk_size)
        chunk_q.put((fr_, to_, cid))
        cid += 1
        fr_ = to_
    assert (rate * fr_ == hfr_lr_in.shape[2])

    # signal workers to move to next stage
    for _ in range(num_workers):
        chunk_q.put(None)
    _rusage('badbp done distributing')

    chunk_q.join()  # wait for all jobs to be submitted
    #print('-badbp- done submitting all chunks to {} workers \t {:.2f} [sec]'.format(num_workers, time.time() - t))

    jid_q.join()  # wait for all frames to be collected
    _rusage('badbp frames collected')

    # signal workers to move to exit
    for _ in range(num_workers):
        jid_q.put(None)
    jid_q.join()  # wait for all threads to get the signal

    # make sure all threads exited
    # assert(all([not w_.is_alive() for w_ in workers]))

    # cleanup
    if err_q.qsize() == 0:
        shutil.rmtree(path=tag, ignore_errors=True)
    else:
        while True:
            try:
                ejid = err_q.get(block=False)
            except Empty:
                break
        err_q.task_done()
        #print('-badbp- job id {} had an error'.format(ejid))
    _rusage('badbp done')
    #print('-badbp- done  \t {:.2f} [sec]'.format(time.time() - t))
    return np.clip(hfr_hr_pred, 0., 1.)


def _read_chunk(folder, max_chunk_size):
    out = []
    for i in range(max_chunk_size):
        try:
            out.append(np.array(Image.open('{}/bp{:05d}.png'.format(folder, i))).astype(np.float32)[..., None, :] / 255.)
        except Exception as err:
            #print('\t\t({}) {}'.format(type(err).__name__, err))
            break
    return np.concatenate(out, axis=2)


def _write_chunk(hwtc_np, folder_name):
    os.makedirs(folder_name)
    for fidx in range(hwtc_np.shape[2]):
        Image.fromarray(np.clip(255*hwtc_np[..., fidx,:], 0, 255).astype(np.uint8)).save('{}/{:06d}.png'.format(folder_name, fidx))


def _submit_job_and_get_jid(pyargs, tag):
    cmd = ['bsub', '-o', '{}%J.o'.format(tag), '-e', '{}%J.e'.format(tag), '-R', 'rusage[mem=8096]', '-q', 'new-short',
           'source /etc/profile.d/modules.sh; module load anaconda/5.2.0/python/2.7; python -u simple_backprojection.py {}'.format(pyargs)]
    #print('-badbp- submitting {}'.format(' '.join(cmd)))
    out = sp.check_output(cmd)
    out = out.decode()
    jid = out.split('<')[1].split('>')[0]
    return jid


def _check_job_status(jid):
    try:
        s = sp.check_output(['bjobs', '{}'.format(jid)])
    except sp.CalledProcessError:
        return 'sperr'
    s = s.decode()
    if len(s) > 10:
        return s.split()[10].lower()
    return 'exit'


def read_frames_from_folder_to_hwtc_np(folder, ext='png'):
    filenames = sorted(glob.glob(os.path.join(folder, '*.{}'.format(ext))))
    frames = []
    for filename in filenames:
        frames.append(np.array(Image.open(filename).convert('RGB')).astype(np.float32)[..., None, :]/255.)
    return np.concatenate(frames, axis=2)


# -------------------------------------------------------------------
# sharpen in time  DO NOT USE IT!!!
# -------------------------------------------------------------------
def temporal_rect3_sharpen_by_backprojection(vid_hwtc, num_iter=2):

    def _rect3t(x_):
        y_ = x_.copy()
        y_[..., 1:-1, :] += (x_[..., 2:, :] + x_[..., :-2, :])
        # handle boundry reflection padding
        y_[..., 0, :] += 2 * x_[..., 1, :]
        y_[..., -1, :] += 2 * x_[..., -2, :]
        return y_ / 3.

    def _tri3t(x_):
        y_ = 0.5 * x_.copy()
        y_[..., 1:-1, :] += 0.25 * (x_[..., 2:, :] + x_[..., :-2, :])
        y_[..., 0, :] += 0.5 * x_[..., 1, :]
        y_[..., -1, :] += 0.5 * x_[..., -2, :]
        return y_
    # spatially chunk it
    chunk_size = 128
    fr_ = [0, 0]
    to_ = [0, 0]

    out = np.empty_like(vid_hwtc)
    for _ in range(num_iter):
        while fr_[0] < out.shape[0]:
            to_[0] = min(fr_[0] + chunk_size, out.shape[0])
            while fr_[1] < out.shape[1]:
                to_[1] = min(fr_[1] + chunk_size, out.shape[1])

                err = vid_hwtc[fr_[0]:to_[0], fr_[1]:to_[1], ...] - _tri3t(vid_hwtc[fr_[0]:to_[0], fr_[1]:to_[1], ...])
                out[fr_[0]:to_[0], fr_[1]:to_[1], ...] = vid_hwtc[fr_[0]:to_[0], fr_[1]:to_[1], ...] + _tri3t(err)

                fr_[1] = to_[1]
            fr_[0] = to_[0]
    return np.clip(out, 0., 1.)


# # -------------------------------------------------------------------
# # DEBUG CODE
# # -------------------------------------------------------------------
def _rusage(tag=''):
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        #print('-DBG- {: >15} CPU mem usage = {:.3f} GiB'.format(tag, usage[2]*resource.getpagesize() / 2.**30))
    except Exception as err:
        z=0
        #print('-DBG- got error ({}) {} while trying to get rusage stats'.format(type(err).__name__, err))


if __name__ == '__main__':
    t = time.time()
    parser = ArgumentParser()
    parser.add_argument('--lfr_hr_in', type=str, help='path to folder where all low frame rate high resolution frames are stored')
    parser.add_argument('--hfr_lr_in', type=str, help='path to folder where all high frame rate low resolution frames are stored')
    parser.add_argument('--hfr_hr_pred', type=str, help='optional: path to folder where all high frame rate low resolution frames are stored', default=None)
    parser.add_argument('--output', type=str, help='output folder')
    parser.add_argument('--ext', type=str, help='file extension for saved frames, default: png', default='png')

    args = parser.parse_args()

    lfr_hr_in = read_frames_from_folder_to_hwtc_np(args.lfr_hr_in, args.ext)
    hfr_lr_in = read_frames_from_folder_to_hwtc_np(args.hfr_lr_in, args.ext)

    hfr_hr_pred = None
    if args.hfr_hr_pred is not None:
        hfr_hr_pred = read_frames_from_folder_to_hwtc_np(args.hfr_hr_pred, args.ext)

    # check inputs
    _check_inputs(hfr_hr_pred=hfr_hr_pred, lfr_hr_in=lfr_hr_in, hfr_lr_in=hfr_lr_in)

    #print('done reading frames {:.3f} [sec]'.format(time.time() - t))

    out = space_time_backprojection_cpu(hfr_hr_pred=hfr_hr_pred, lfr_hr_in=lfr_hr_in, hfr_lr_in=hfr_lr_in)

    #print('done backprojection {:.3f} [sec]'.format(time.time() - t))

    try:
        os.makedirs(args.output)
    except Exception as err:
        z=0
        #print('got error ({}) {} while creating folder. ignoring.'.format(type(err).__name__, err))

    for i in range(out.shape[2]):
        Image.fromarray(
            np.clip(255*out[..., i, :], 0, 255).astype(np.uint8)
        ).save(os.path.join(args.output, 'bp{:05d}.png'.format(i)))

    #print('done. {:.3f} [sec]'.format(time.time() - t))

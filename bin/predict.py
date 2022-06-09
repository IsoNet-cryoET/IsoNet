#!/usr/bin/env python3
import os, sys
from IsoNet.util.image import *
from IsoNet.util.metadata import MetaData,Label,Item
from IsoNet.util.dict2attr import idx2list
import logging

def predict(args):

    logger = logging.getLogger('predict')
    if args.log_level == "debug":
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt="%H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])
    else:
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
        datefmt="%m-%d %H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])
    logging.info('\n\n######Isonet starts predicting######\n')

    args.gpuID = str(args.gpuID)
    args.ngpus = len(list(set(args.gpuID.split(','))))
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpuID
    logger.info('percentile:{}'.format(args.normalize_percentile))
    logger.info('gpuID:{}'.format(args.gpuID))

    from IsoNet.models.network import Net
    network = Net()
    network.load(args.model)

    if args.batch_size is None:
        args.batch_size = 4 * args.ngpus
    
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    md = MetaData()
    md.read(args.star_file)
    if not 'rlnCorrectedTomoName' in md.getLabels():
        md.addLabels('rlnCorrectedTomoName')
        for it in md:
            md._setItemValue(it,Label('rlnCorrectedTomoName'),None)
    args.tomo_idx = idx2list(args.tomo_idx)

    for it in md:
        if args.tomo_idx is None or str(it.rlnIndex) in args.tomo_idx:
            if args.use_deconv_tomo and "rlnDeconvTomoName" in md.getLabels() and it.rlnDeconvTomoName not in [None,'None']:
                tomo_file = it.rlnDeconvTomoName
            else:
                tomo_file = it.rlnMicrographName
            tomo_root_name = os.path.splitext(os.path.basename(tomo_file))[0]
            if os.path.isfile(tomo_file):
                tomo_out_name = '{}/{}_corrected.mrc'.format(args.output_dir,tomo_root_name)
                network.predict_tomo(args,tomo_file,output_file=tomo_out_name)
                md._setItemValue(it,Label('rlnCorrectedTomoName'),tomo_out_name)
        md.write(args.star_file)


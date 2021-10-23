import numpy as np
import torch
import torch.nn.functional as F
from metrics import distance, rank

# ---------------------- Extract features ----------------------


def _parse_data_for_eval(data):
    imgs = data[0]
    pids = data[1]
    camids = data[2]
    return imgs, pids, camids


def _extract_features(model, input):
    model.eval()
    return model(input)


# devie-------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------- Start testing ----------------------
@torch.no_grad()
def test(model, test_loader, args, normalize_feature=False, dist_metric='cosine'):
    model.eval()

    # test dataloader------------------------------------------------------------
    query_dataloader, gallery_dataloader = test_loader

    # Extracting features from query set------------------------------------------------------------
    print('Extracting features from query set ...')
    qf, q_pids, q_camids = [], [], []  # query features, query person IDs and query camera IDs
    q_score = []
    for batch_idx, data in enumerate(query_dataloader):
        imgs, pids, camids = _parse_data_for_eval(data)
        imgs = imgs.to(device)
        features = _extract_features(model, imgs)
        qf.append(features)
        q_pids.extend(pids)
        q_camids.extend(camids)
    qf = torch.cat(qf, 0)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

    # Extracting features from gallery set------------------------------------------------------------
    print('Extracting features from gallery set ...')
    gf, g_pids, g_camids = [], [], []  # gallery features, gallery person IDs and gallery camera IDs
    g_score = []
    for batch_idx, data in enumerate(gallery_dataloader):
        imgs, pids, camids = _parse_data_for_eval(data)
        imgs = imgs.to(device)
        features = _extract_features(model, imgs)
        gf.append(features)
        g_pids.extend(pids)
        g_camids.extend(camids)
    gf = torch.cat(gf, 0)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

    # normalize_feature------------------------------------------------------------------------------
    if normalize_feature:
        print('Normalzing features with L2 norm ...')
        qf = F.normalize(qf, p=2, dim=1)
        gf = F.normalize(gf, p=2, dim=1)

    # Computing distance matrix------------------------------------------------------------------------
    print('Computing distance matrix with metric={} ...'.format(dist_metric))
    qf = np.array(qf.cpu())
    gf = np.array(gf.cpu())
    dist = distance.cosine_dist(qf, gf)
    rank_results = np.argsort(dist)[:, ::-1]

    # Computing CMC and mAP------------------------------------------------------------------------
    print('Computing CMC and mAP ...')
    APs, CMC = [], []
    for idx, data in enumerate(zip(rank_results, q_camids, q_pids)):
        a_rank, query_camid, query_pid = data
        ap, cmc = rank.compute_AP(a_rank, query_camid, query_pid, g_camids, g_pids)
        APs.append(ap), CMC.append(cmc)
    MAP = np.array(APs).mean()
    min_len = min([len(cmc) for cmc in CMC])
    CMC = [cmc[:min_len] for cmc in CMC]
    CMC = np.mean(np.array(CMC), axis=0)

    return CMC, MAP


# if __name__ == "__main__":
#     # -----------------------------parameters setting --------------------------------
#     parser = argparse.ArgumentParser(description='Testing arguments')

#     parser.add_argument('--experiment', type=str, default='pcb_rga_v5')
#     parser.add_argument('--save_path', type=str, default='../experiments')
#     parser.add_argument('--which_epoch', default='final', type=str, help='0,1,2,3...or final')
#     parser.add_argument('--checkpoint', type=str, default='/home/hy/vscode/reid-custom/experiments/Market1501')

#     parser.add_argument('--dataset', type=str, default='Occluded_REID')
#     parser.add_argument('--dataset_path', type=str, default='/home/hy/vscode/data/Occluded_REID')
#     parser.add_argument('--height', type=int, default=384, help='height of the input image')
#     parser.add_argument('--width', type=int, default=128, help='width of the input image')

#     parser.add_argument('--batch_size', default=3, type=int, help='batchsize')
#     parser.add_argument('--share_conv', default=False, action='store_true')

#     args = parser.parse_args()

#     # speed up compution---------------------------------------------------------------------------
#     cudnn.benchmark = True

#     # devie---------------------------------------------------------------------------
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # model------------------------------------------------------------------------------------
#     model = build_model(args.experiment, num_classes=1, height=args.height, width=args.width)
#     model = util.load_network(model, args.checkpoint, args.which_epoch)
#     model = model.to(device)

#     # save_dir_path-----------------------------------------------------------------------------------
#     save_dir_path = os.path.join(args.save_path, args.dataset)
#     os.makedirs(save_dir_path, exist_ok=True)

#     # logger------------------------------------------------------------------------------------
#     logger = util.Logger(save_dir_path)
#     logger.info(vars(args))

#     # test -----------------------------------------------------------------------------------
#     CMC, mAP = test(model, args.dataset, args.dataset_path, args.batch_size, device, args)
#     logger.info('Testing: top1:%.4f top5:%.4f top10:%.4f mAP:%.4f' % (CMC[0], CMC[4], CMC[9], mAP))

#     # torch.cuda.empty_cache()

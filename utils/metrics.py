import torch
import os.path as osp
import numpy as np
import os
from utils.reranking import re_ranking
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import ImageDraw, Image
def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img
def plot_rank(qimg_path, gimg_paths, order, matches, save_dir = "", epoch = - 1):
    num_figs = min(5, len(order)) #plot at most top 5 answers
    gimg_paths = gimg_paths[order]
    gimg_paths = gimg_paths[:num_figs]
    import pdb
    pdb.set_trace()
    rank =5  #top "rank" images are plotted
    fig = plt.figure(figsize=(30, 20))
    a = fig.add_subplot(1, rank + 1, 1)
    a.set_title("QUERY")
    a.axis("off")
    img = read_image(qimg_path)
    img = img.resize((128, 128), Image.BILINEAR)
    plt.imshow(img)
    for ind, gimg_path in enumerate(gimg_paths):
        a = fig.add_subplot(1, rank + 1, ind + 2)
        message = "CORRECT" if matches[ind] == 1 else "WRONG"
        color = "green" if matches[ind] == 1 else "red"
#        color = "blue"
        a.set_title(message)
        a.axis("off")
        img = read_image(gimg_path)
        img = img.resize((128, 128), Image.BILINEAR)
        draw = ImageDraw.Draw(img)
        draw.rectangle([(0, 0), (img.size[0] - 1, img.size[1] - 1)], outline = color)
        draw.rectangle([(1, 1), (img.size[0] - 2, img.size[1] - 2)], outline = color)
        draw.rectangle([(2, 2), (img.size[0] - 3, img.size[1] - 3)], outline = color)
        plt.imshow(img)
        del draw
    save_dir=osp.join(save_dir,'{}'.format(epoch))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    q_imgname = qimg_path.split('/')[-1]
    fig.savefig(os.path.join(save_dir, q_imgname), bbox_inches='tight')
    fig.clf()
    plt.close()
    print("plot rank done")
    del a
#    gc.collect()





def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(cfg,distmat, q_pids, g_pids,q_cids,g_cids, q_camids, g_camids, max_rank=50,query_img_paths = [], gallery_img_paths = [],save_dir = "", epoch = -1, save_rank = False):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_cid = q_cids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        if cfg.MODEL.STANDARD_METRIC:
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        if cfg.MODEL.CLOTH_METRIC:
#            print('use-cloth-changing-metric')
         #   remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            remove = (g_pids[order] == q_pid) & ((g_camids[order] == q_camid) | (g_cids[order] == q_cid))
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        if save_rank:
            #plot rank results
            plot_rank(query_img_paths[q_idx], gallery_img_paths, order[keep], matches[q_idx][keep], save_dir, epoch)

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP_eval():
    def __init__(self,cfg, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.cfg=cfg
    def reset(self):
        self.feats = []
        self.pids = []
        self.cids = []
        self.camids = []
        self.img_paths = []
    def update(self, output):  # called once for each batch
        feat, pid,cid, camid,img_paths = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.cids.extend(np.asarray(cid))
        self.camids.extend(np.asarray(camid))
        self.img_paths.extend(np.asarray(img_paths))
    def vis(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        # query
        pids = np.asarray(self.pids)
        tsne = TSNE(n_components=2, random_state=0)
        labels=pids
        data_2d = tsne.fit_transform(feats)
        
    # Create a scatter plot of the 2D data
        plt.figure(figsize=(8, 6))
        markers = ['o', 's', '^', 'v', '*', 'p', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
        num_classes = len(np.unique(labels))
        colors = plt.cm.get_cmap('tab10', num_classes)

    # Create a scatter plot of the 2D data
        for i, label in enumerate(np.unique(labels)):
            class_data = data_2d[labels == label]
            class_labels = labels[labels == label]
            color = [colors(i) for _ in range(len(class_data))] 
            plt.scatter(class_data[:, 0], class_data[:, 1], c=color,  s=50, marker=markers[i % len(markers)], label=f'Class {label}')
#        scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=pids, cmap='tab10', s=10, alpha=0.7)
     #   import pdb
     #   pdb.set_trace()
    # Add a legend
#        plt.legend(*scatter.legend_elements(), title="Classes")
        #plt.legend()
#        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
#        name="Baseline+SGR+SRA"
#        name="Baseline"
#        name="Baseline+SGR+SRA+PRS"
        name="Baseline+SSG+PARC"
       # plt.title(name)
            # 移除刻度
        plt.xticks([])
        plt.yticks([])
#        ax.set_aspect('equal')
        plt.xlabel(name)
 #       plt.xlim(-50, 50)
 #       plt.ylim(-50, 50)
       # plt.ylabel('t-SNE Component 2')
        # compute(self):  # called after each epoch
        plt.savefig('/home/rxn/myproject/clip_rl_uav/fig/'+name+'.png', dpi=300, bbox_inches='tight')
        #plt.savefig('/home/rxn/myproject/clip_rl_uav/fig/'+name+'.png', dpi=300, bbox_inches='tight')
        print("plot figure done")







    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_cids = np.asarray(self.cids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_img_paths = np.asarray(self.img_paths[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_cids = np.asarray(self.cids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        g_img_paths = np.asarray(self.img_paths[self.num_query:])

        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
#        import pdb
#        pdb.set_trace()
#        cmc, mAP = eval_func(self.cfg,distmat, q_pids, g_pids,q_cids,g_cids, q_camids, g_camids)
        print(distmat.shape)
        print(distmat.shape)
        print(distmat.shape)
        print(distmat.shape)
        print("******************************")
        cmc, mAP = eval_func(self.cfg,distmat, q_pids, g_pids,q_cids,g_cids, q_camids, g_camids,self.max_rank,q_img_paths,g_img_paths,save_dir = '/home/cncert/cliptest/ranklist', epoch = 200, save_rank =True )

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf




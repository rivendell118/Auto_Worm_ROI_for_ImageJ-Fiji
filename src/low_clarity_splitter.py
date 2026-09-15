"""Conservative separation of parallel worms joined by uncertain interior bridges.

Only the low-clarity inference path calls this module. Counts are not targets:
two long interior cores, a head-end gap and a sustained boundary valley must
agree before an existing instance may be divided. No foreground is added.
"""
from dataclasses import dataclass
import heapq

import numpy as np
from scipy import ndimage


@dataclass
class SplitReport:
    original_label: int
    status: str
    threshold: float
    head_gap: float
    boundary_support: float
    overlap_fraction: float
    child_area_ratio: float
    resulting_labels: str = ""


def _core_evidence(first, second, probability, image, parent_height):
    y1, x1 = np.nonzero(first)
    y2, x2 = np.nonzero(second)
    if x1.mean() > x2.mean():
        first, second = second, first
        y1, x1, y2, x2 = y2, x2, y1, x1
    rows = np.arange(max(y1.min(), y2.min()), min(y1.max(), y2.max()) + 1)
    shared = [int(y) for y in rows if first[y].any() and second[y].any()]
    overlap = len(shared) / max(parent_height, 1)
    if overlap < .45:
        return None
    centers = np.asarray([(y, np.flatnonzero(first[y]).mean(),
                           np.flatnonzero(second[y]).mean()) for y in shared])
    spacing = centers[:, 2] - centers[:, 1]
    if np.percentile(spacing, 5) < 6:
        return None
    # A long valley between simultaneous cores distinguishes parallel worms
    # from two fragments arranged along a single worm's length.
    supports = []
    for y, left, right in centers:
        y = int(y)
        lo, hi = int(left + .25 * (right-left)), int(right - .25 * (right-left))
        valley = probability[2, y, lo:hi+1] + probability[0, y, lo:hi+1]
        supports.append(float(valley.max()))
    support = float(np.mean(np.asarray(supports) >= .45))
    if support < .55:
        return None
    # Follow the first simultaneous core centers toward the head tips. Require
    # exterior background in the gap plus brighter material on BOTH sides.
    # A dark internal organ alone cannot satisfy the background test.
    head_scores = []
    start = int(centers[0, 0])
    fit = centers[:max(8, int(.08 * parent_height))]
    left_fit = np.polyfit(fit[:, 0] - start, fit[:, 1], 1)
    right_fit = np.polyfit(fit[:, 0] - start, fit[:, 2], 1)
    stop = min(image.shape[0], start + max(5, int(.06 * parent_height)))
    for y in range(max(0, start - int(.20 * parent_height)), stop):
        left = (float(np.polyval(left_fit, y-start)) if y < start
                else float(np.interp(y, centers[:, 0], centers[:, 1])))
        right = (float(np.polyval(right_fit, y-start)) if y < start
                 else float(np.interp(y, centers[:, 0], centers[:, 2])))
        if not (2 <= left < right - 6 and right < image.shape[1]-2):
            continue
        gap_lo = int(round(left + .2 * (right-left)))
        gap_hi = int(round(right - .2 * (right-left)))
        if gap_hi <= gap_lo:
            continue
        x = gap_lo + int(np.argmax(probability[0, y, gap_lo:gap_hi+1]))
        # Use a band rather than a single pixel to tolerate blurred tip positions.
        radius = max(2, int(.22 * (right-left)))
        l0,l1=max(0,int(left)-radius),min(image.shape[1],int(left)+radius+1)
        r0,r1=max(0,int(right)-radius),min(image.shape[1],int(right)+radius+1)
        flank = min(float(image[y,l0:l1].max()), float(image[y,r0:r1].max()))
        contrast = flank - float(image[y,x])
        side_foreground = min(float((1-probability[0,y,l0:l1]).max()),
                              float((1-probability[0,y,r0:r1]).max()))
        if contrast >= .025 and side_foreground >= .60:
            head_scores.append((y,float(probability[0,y,x])))
    # Heads can be staggered: a short contiguous gap is meaningful even when
    # most of the searched band already lies inside both bodies.
    windows = [float(np.mean([v for _,v in head_scores[i:i+5]]))
               for i in range(len(head_scores)-4)
               if head_scores[i+4][0]-head_scores[i][0] == 4]
    head_gap = max(windows,default=0.)
    if head_gap < .30:
        return None
    return first, second, head_gap, support, overlap


def _grow_cores(mask, cores, probability):
    """Multi-source geodesic growth confined to the original foreground union."""
    height, width = mask.shape
    owners = np.zeros(mask.shape, dtype=np.uint8)
    distance = np.full(mask.shape, np.inf)
    cost = 1. + 8. * (probability[0] + probability[2])
    queue = []
    for label, core in enumerate(cores, 1):
        owners[core] = label
        distance[core] = 0.
        edge = core & ~ndimage.binary_erosion(core)
        for y, x in np.argwhere(edge):
            heapq.heappush(queue, (0., int(y), int(x), label))
    while queue:
        value, y, x, owner = heapq.heappop(queue)
        if value != distance[y,x] or owner != owners[y,x]:
            continue
        for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
            ny,nx=y+dy,x+dx
            if not (0 <= ny < height and 0 <= nx < width and mask[ny,nx]):
                continue
            proposed = value + .5 * (cost[y,x] + cost[ny,nx])
            if proposed < distance[ny,nx]:
                distance[ny,nx]=proposed
                owners[ny,nx]=owner
                heapq.heappush(queue,(proposed,ny,nx,owner))
    return owners


def _ordered_partition(mask, cores, probability):
    """Track ordered longitudinal seams through jointly affected neighbours.

    Short high-confidence cores cannot own a complete tail by nearest-distance
    growth alone. Row-wise center fractions supply an end continuation prior;
    dynamic programming follows model boundary evidence inside that corridor.
    """
    yy,xx=np.nonzero(mask)
    y0,y1=int(yy.min()),int(yy.max())+1
    height,width=mask.shape
    rows=np.arange(y0,y1)
    left=np.asarray([np.flatnonzero(mask[y])[0] if mask[y].any() else np.nan for y in rows])
    right=np.asarray([np.flatnonzero(mask[y])[-1] if mask[y].any() else np.nan for y in rows])
    valid=np.isfinite(left)
    left=np.interp(rows,rows[valid],left[valid])
    right=np.interp(rows,rows[valid],right[valid])
    span=np.maximum(right-left,1.)
    ordered=sorted(cores,key=lambda core:np.nonzero(core)[1].mean())
    fractions=[]
    for core in ordered:
        cy,cx=np.nonzero(core)
        core_rows=np.unique(cy)
        centers=np.asarray([np.flatnonzero(core[y]).mean() for y in core_rows])
        values=(centers-left[core_rows-y0])/span[core_rows-y0]
        # Use only the reliable body for extending beyond a shortened core.
        reliable=(core_rows>=y0+.15*(y1-y0)) & (core_rows<=y0+.78*(y1-y0))
        if reliable.sum()>=8:
            core_rows,values=core_rows[reliable],values[reliable]
        fraction=ndimage.gaussian_filter1d(np.interp(rows,core_rows,values),sigma=3.)
        fractions.append(fraction)
    curves=left[None,:]+np.asarray(fractions)*span[None,:]
    if np.any(np.diff(curves,axis=0)<=0.):
        return None
    seams=[]
    xs=np.arange(width)
    for index in range(len(ordered)-1):
        a,b=curves[index],curves[index+1]
        middle=(a+b)*.5
        half=np.maximum((b-a)*.35,2.)
        # A soft corridor tolerates local diagonal tails without an impossible
        # hard-constrained path. Bright interior incurs a high crossing cost.
        unary=probability[1,y0:y1].astype(np.float64)*3.
        unary+=.22*((xs[None,:]-middle[:,None])/half[:,None])**2
        unary+=np.maximum(np.abs(xs[None,:]-middle[:,None])-half[:,None],0.)*2.
        score=unary[0].copy()
        parents=np.zeros(unary.shape,dtype=np.int8)
        for row in range(1,len(rows)):
            transitions=np.full((7,width),np.inf)
            for j,step in enumerate(range(-3,4)):
                src=xs+step
                valid=(src>=0)&(src<width)
                transitions[j,valid]=score[src[valid]]+.06*abs(step)
            best=np.argmin(transitions,axis=0)
            parents[row]=best-3
            score=unary[row]+transitions[best,xs]
        seam=np.empty(len(rows),dtype=np.int32)
        seam[-1]=int(np.argmin(score))
        for row in range(len(rows)-1,0,-1):
            seam[row-1]=seam[row]+parents[row,seam[row]]
        seams.append(seam)
    if len(seams)>1 and np.any(np.diff(np.asarray(seams),axis=0)<=0):
        return None
    owner=np.ones((len(rows),width),dtype=np.uint16)
    for seam in seams:
        owner+=(xs[None,:]>seam[:,None])
    result=np.zeros(mask.shape,dtype=np.uint16)
    result[y0:y1]=np.where(mask[y0:y1],owner,0)
    head_end=max(int(np.nonzero(core)[0].min()) for core in ordered)
    head_partition=_grow_cores(mask,ordered,probability)
    result[:head_end]=head_partition[:head_end]
    # A seam must not leave an isolated sliver assigned to the wrong child.
    # Reattach only small pre-existing fringe pieces to a touching neighbour.
    for i in range(1,len(ordered)+1):
        components,n=ndimage.label(result==i)
        if n<=1:
            continue
        sizes=np.bincount(components.ravel());sizes[0]=0
        main=int(np.argmax(sizes))
        for part in range(1,n+1):
            if part==main or sizes[part]>.03*(result==i).sum():
                continue
            fragment=components==part
            adjacent=result[ndimage.binary_dilation(fragment)&~fragment]
            adjacent=adjacent[(adjacent>0)&(adjacent!=i)]
            if len(adjacent):
                result[fragment]=int(np.bincount(adjacent).argmax())
    missing=mask&(result==0)
    if missing.any():
        nearest=ndimage.distance_transform_edt(result==0,return_distances=False,return_indices=True)
        result[missing]=result[tuple(nearest)][missing]
    for i,core in enumerate(ordered,1):
        region=result==i
        components,n=ndimage.label(region)
        if not n or np.bincount(components.ravel())[1:].max()<.995*region.sum():
            return None
        if (region & core).sum()<.95*core.sum():
            return None
    return result


def split_low_clarity_instances(instances, probabilities, image, max_instances=24):
    """Return labels and accepted-split reports; unchanged inputs remain exact."""
    original = np.asarray(instances)
    if probabilities.shape != (3, *original.shape) or image.shape != original.shape:
        raise ValueError('Split probability/image dimensions do not match instances')
    if not np.isfinite(probabilities).all() or not np.isfinite(image).all():
        raise ValueError('Split evidence contains non-finite values')
    output = original.copy()
    labels = [int(v) for v in np.unique(original) if v]
    if len(labels) < 3:
        return output, []
    measurements = []
    for label in labels:
        yy,xx=np.nonzero(original==label)
        measurements.append((label,len(yy),int(np.ptp(yy))+1,len(yy)/(np.ptp(yy)+1)))
    normal_width = float(np.median([m[3] for m in measurements]))
    normal_area = float(np.median([m[1] for m in measurements]))
    reports=[]
    accepted=[]
    next_label=max(labels)+1
    for label, area, height, mean_width in measurements:
        if max_instances and len(labels)+len(reports) >= max_instances:
            break
        if mean_width < 1.35*normal_width or area < 1.30*normal_area:
            continue
        yy,xx=np.nonzero(original==label)
        y0,y1=max(0,int(yy.min())-2),min(original.shape[0],int(yy.max())+3)
        x0,x1=max(0,int(xx.min())-2),min(original.shape[1],int(xx.max())+3)
        mask=original[y0:y1,x0:x1]==label
        p=probabilities[:,y0:y1,x0:x1]
        gray=image[y0:y1,x0:x1]
        for threshold in (.55,.70,.85,.95):
            components, number=ndimage.label(ndimage.binary_erosion(
                mask & (p[1]>=threshold),iterations=2))
            cores=[]
            for component in range(1,number+1):
                core=components==component
                cy,cx=np.nonzero(core)
                if len(cy)>=max(.08*area,.20*normal_area) and np.ptp(cy)>=.50*height:
                    cores.append(core)
            if len(cores)!=2:
                continue
            evidence=_core_evidence(*cores,p,gray,height)
            if evidence is None:
                continue
            first,second,gap,support,overlap=evidence
            partition=_grow_cores(mask,(first,second),p)
            stranded = mask & (partition==0)
            if stranded.sum() > .002 * area:
                continue
            if stranded.any():
                nearest=ndimage.distance_transform_edt(
                    partition==0,return_distances=False,return_indices=True)
                partition[stranded]=partition[tuple(nearest)][stranded]
            areas=[int((partition==i).sum()) for i in (1,2)]
            if min(areas)<.45*normal_area or min(areas)/max(areas)<.30:
                continue
            connected=True
            for i in (1,2):
                cc,n=ndimage.label(partition==i)
                if n and np.bincount(cc.ravel())[1:].max() < .998*areas[i-1]:
                    connected=False
            if not connected:
                continue
            local=output[y0:y1,x0:x1]
            local[partition==1]=label
            local[partition==2]=next_label
            next_label+=1
            reports.append(SplitReport(label,'REVIEW_LOW_CLARITY_SPLIT',threshold,
                                       gap,support,overlap,min(areas)/max(areas)))
            global_cores=[]
            for core in (first,second):
                full=np.zeros(original.shape,dtype=bool)
                full[y0:y1,x0:x1]=core
                global_cores.append(full)
            accepted.append((label,global_cores))
            break
    if reports:
        # Consecutive merged instances may have exchanged tail pixels in the
        # coarse nearest-interior assignment; partition their union together.
        groups=[]
        for item in accepted:
            if groups and labels.index(item[0])==labels.index(groups[-1][-1][0])+1:
                groups[-1].append(item)
            else:
                groups.append([item])
        output=original.copy()
        kept=[]
        final_pairs={}
        next_label=max(labels)+1
        for group in groups:
            region=np.isin(original,[label for label,_ in group])
            cores=[core for _,parts in group for core in parts]
            partition=_ordered_partition(region,cores,probabilities)
            if partition is None:
                continue
            ids=[label for label,_ in group]+list(range(next_label,next_label+len(group)))
            next_label+=len(group)
            for i,label in enumerate(ids,1):
                output[partition==i]=label
            for j,(parent,_) in enumerate(group):
                report=next(r for r in reports if r.original_label==parent)
                child_areas=[int((partition==k).sum()) for k in (2*j+1,2*j+2)]
                report.child_area_ratio=min(child_areas)/max(child_areas)
                final_pairs[parent]=(ids[2*j],ids[2*j+1])
                kept.append(report)
        reports=kept
        if not reports:
            return original.copy(), []
        # Preserve the established left-to-right ROI numbering convention.
        order=sorted(np.unique(output[output>0]),
                     key=lambda label: np.nonzero(output==label)[1].mean())
        lookup=np.zeros(int(output.max())+1,dtype=np.uint16)
        for index,label in enumerate(order,1):
            lookup[label]=index
        output=lookup[output]
        for report in reports:
            report.resulting_labels="|".join(str(int(lookup[label]))
                                             for label in final_pairs[report.original_label])
    return output, reports

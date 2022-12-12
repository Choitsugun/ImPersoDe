from hyperparams import set_args
import linecache
import codecs
import re

def prof_opt():
    # ======= deduplication prof ======= #
    f = codecs.open('../dataset/train/prof.txt','r','utf-8')
    ft = codecs.open('../dataset/analysis/prof-al.txt','w','utf-8')
    Split = " ===== "
    plist = []
    count = 0

    for line in f:
        line = line.strip()
        line = line.split(Split)
        assert len(line) == 2
        line0 = line[0]
        line1 = line[1]

        flag_ctu0 = True
        flag_ctu1 = True
        flag0 = True
        flag1 = True

        for index, element in enumerate(plist):
            set0 = set(line0.split())
            set1 = set(line1.split())
            sete = set(element.split())
            intersect0 = len(set0) + len(sete) - len(set.union(set0, sete))
            intersect1 = len(set1) + len(sete) - len(set.union(set1, sete))

            if (intersect0/len(set0))>args.prof_ol or (intersect0/len(sete))>args.prof_ol and flag_ctu0 is True:
                flag0 = False
                flag_ctu0 = False
                if len(set0) > len(sete):
                    plist[index] = line0

            if (intersect1/len(set1))>args.prof_ol or (intersect1/len(sete))>args.prof_ol and flag_ctu1 is True:
                flag1 = False
                flag_ctu1 = False
                if len(set1) > len(sete):
                    plist[index] = line1

            if flag_ctu0 is False and flag_ctu1 is False:
                break

        if flag0 is True:
            plist.append(line0)

        if flag1 is True:
            plist.append(line1)

        count = count + 1
        print(count)

    # ================ all prof ================ #
    """
    f = codecs.open('../dataset/train/prof.txt','r','utf-8')
    ft = codecs.open('../dataset/analysis/prof-al.txt','w','utf-8')
    Split = " ===== "
    plist = []

    for line in f:
        line = line.strip()
        line = line.split(Split)
        assert len(line) == 2
        line0 = line[0]
        line1 = line[1]

        if line0 not in plist:
            plist.append(line0)

        if line1 not in plist:
            plist.append(line1)
    """

    # ============= resp in the prof ============ #
    """
    f = codecs.open('../dataset/train/prof-train.txt','r','utf-8')
    file_path = '../dataset/train/resp-train.txt'
    ft = codecs.open('../dataset/analysis/prof-al.txt','w','utf-8')
    plist = []
    rline_number = 0

    for line in f:
        line = line.strip()

        rline_number += 1
        rline = linecache.getline(file_path, rline_number)
        rline = rline.strip()
        rlist = rline.split()

        for pat in rlist:
            if re.search(pat, line) is not None:
                if line not in plist:
                    plist.append(line)
                break
    print(rline_number)
    """

    for i in plist:
        ft.write(i + '\n')
    ft.close()
    f.close()


if __name__ == '__main__':
    args = set_args()
    prof_opt()


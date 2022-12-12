from hyperparams import set_args
import linecache
import codecs
import re

Tab = '\t'
Split = " ===== "

def high_frequency_judgment(line):
    r = line.split(Tab)[-1]
    r_list = r.split()
    w = r_list[0]
    if (re.match("i",w) is None) and (re.match("my",w) is None):
        return True
    elif ("you" in r_list) or ("your" in r_list):
        return True
    else:
        return False


def write_truncate_process(d_line, p_list, dtf, ptf, i, tr_count):
    count_flag = True

    while True:
        if len(d_line.split()) <= args.max_seq_len:
            dtf.write(d_line + '\n')
            if (i % 2) == 1:
                ptf.write(str(p_list[0]) + '\n')
            else:
                ptf.write(str(p_list[1]) + '\n')
            break
        else:
            d_line_list = d_line.split(Tab)[1:]
            if len(d_line_list) >= args.min_utt_turn:
                d_line = Tab.join(d_line_list)
                if count_flag == True:
                    count_flag = False
                    tr_count += 1
            else:
                print("Can not Truncated")
                return tr_count

    return tr_count


def process_step1():
    f = codecs.open('../dataset/agent/train/train.txt', 'r', 'utf-8')
    df = codecs.open('../dataset/agent/train/dial.txt', 'w', 'utf-8')
    pf = codecs.open('../dataset/agent/train/prof.txt', 'w', 'utf-8')

    yp_line = ""
    pp_line = ""
    d_line = ""
    flag = False

    for line in f:
        if re.match("your persona:", line) is not None:
            if flag is True:
                df.write(d_line + '\n')
                pf.write(yp_line + Split + pp_line + '\n')
                yp_line = ""
                pp_line = ""
                d_line = ""
                flag = False
            yp = line.replace("your persona:", "", 1).strip()
            if yp_line == "":
                yp_line = yp
            else:
                yp_line = yp_line + Tab + yp
        elif re.match("partner's persona:", line) is not None:
            pp = line.replace("partner's persona:", "", 1).strip()
            if pp_line == "":
                pp_line = pp
            else:
                pp_line = pp_line + Tab + pp
        else:
            if d_line == "":
                d_line = line.strip()
                flag = True
            else:
                d_line = d_line + Tab + line.strip()

    df.write(d_line + '\n')
    pf.write(yp_line + Split + pp_line + '\n')
    df.close()
    pf.close()


def process_step2():
    df = codecs.open('../dataset/agent/train/dial.txt', 'r', 'utf-8')
    file_path = '../dataset/agent/train/prof.txt'
    dtf = codecs.open('../dataset/agent/train/dial-train.txt', 'w', 'utf-8')
    ptf = codecs.open('../dataset/agent/train/prof-train.txt', 'w', 'utf-8')
    p_line_number = 0
    tr_count = 0
    re_count = 0

    for line in df:
        d_line = line.strip()
        d_list = d_line.split(Tab)
        turn_len = len(d_list)

        p_line_number += 1
        p_line = linecache.getline(file_path, p_line_number)
        p_line = p_line.strip()
        p_list = p_line.split(Split)

        if turn_len < args.min_utt_turn:
            print("turn of dialogue is too short")
        else:
            for i in range(args.min_utt_turn, turn_len + 1):
                d_arry = d_list[:i]
                d_line = Tab.join(d_arry)

                if len(d_arry[-1].split()) < args.min_resp_len:
                    re_count = re_count + 1
                    print("resp is too short")
                    continue

                if high_frequency_judgment(d_line):
                    tr_count = write_truncate_process(d_line, p_list, dtf, ptf, i, tr_count)

    print(p_line_number)
    print(tr_count)
    print(re_count)
    dtf.close()
    ptf.close()


def process_step3():
    dtf = codecs.open('../dataset/agent/train/dial-train.txt', 'r', 'utf-8')
    ctf = codecs.open('../dataset/agent/train/cont-train.txt', 'w', 'utf-8')
    rtf = codecs.open('../dataset/agent/train/resp-train.txt', 'w', 'utf-8')

    for line in dtf:
        d_list = line.split(Tab)
        r = d_list[-1]
        c = Tab.join(d_list[:-1])
        ctf.write(c + '\n')
        rtf.write(r)

    ctf.close()
    rtf.close()


if __name__ == '__main__':
    args = set_args()
    process_step1()
    process_step2()
    process_step3()
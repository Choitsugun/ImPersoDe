from hyperparams import set_args
import linecache
import codecs
import re

Tab = '\t'
Split = " ===== "

def write_speci_turn_process(d_line, p_list, dtf, ptf):
    if len(d_line.split(Tab)) < args.spe_utt_turn:
        return

    while True:
        if len(d_line.split(Tab)) == args.spe_utt_turn:
            if len(d_line.split()) > args.max_seq_len:
                print("seq of dialogue is too long")
                break

            dtf.write(d_line + '\n')

            # P Tab I Tab P Tab I ---> dial.txt
            # I === P ---> prof.txt
            if (len(d_line.split(Tab)) % 2) == 1:
                ptf.write(str(p_list[0]) + '\n')
            else:
                ptf.write(str(p_list[1]) + '\n')
            break
        else:
            d_line_list = d_line.split(Tab)[:args.spe_utt_turn]
            d_line = Tab.join(d_line_list)


def process_step1():
    f = codecs.open('../dataset/agent/test/test.txt', 'r', 'utf-8')
    df = codecs.open('../dataset/agent/test/dial.txt', 'w', 'utf-8')
    pf = codecs.open('../dataset/agent/test/prof.txt', 'w', 'utf-8')

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
    df = codecs.open('../dataset/agent/test/dial.txt', 'r', 'utf-8')
    file_path = '../dataset/agent/test/prof.txt'
    dtf = codecs.open('../dataset/agent/test/dial-test.txt', 'w', 'utf-8')
    ptf = codecs.open('../dataset/agent/test/prof-test.txt', 'w', 'utf-8')
    p_line_number = 0

    for line in df:
        d_line = line.strip()

        p_line_number += 1
        p_line = linecache.getline(file_path, p_line_number)
        p_line = p_line.strip()
        p_list = p_line.split(Split)

        write_speci_turn_process(d_line, p_list, dtf, ptf)

    print(p_line_number)
    dtf.close()
    ptf.close()

    return


def process_step3():
    dtf = codecs.open('../dataset/agent/test/dial-test.txt', 'r', 'utf-8')
    ctf = codecs.open('../dataset/agent/test/cont-test.txt', 'w', 'utf-8')
    rtf = codecs.open('../dataset/agent/test/resp-test.txt', 'w', 'utf-8')

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
from pathlib import Path
import csv
import json
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import pandas as pd

try:  # import khi chạy test (scripts.<x>) hoặc chạy trực tiếp (python scripts/<x>.py)
    from scripts.provenance import DATA_ROOT
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from provenance import DATA_ROOT

ROOT = DATA_ROOT  # thư mục dữ liệu/kết quả (KLTN_DATA_ROOT)

def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr(); shd = OxmlElement('w:shd'); shd.set(qn('w:fill'), color); tcPr.append(shd)

def style_doc(d):
    sec=d.sections[0]; sec.top_margin=Inches(.7); sec.bottom_margin=Inches(.7); sec.left_margin=Inches(.75); sec.right_margin=Inches(.75)
    styles=d.styles; styles['Normal'].font.name='Arial'; styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'),'Arial'); styles['Normal'].font.size=Pt(10)
    for name,size in [('Title',18),('Heading 1',14),('Heading 2',11)]:
        styles[name].font.name='Arial'; styles[name].font.size=Pt(size); styles[name].font.color.rgb=RGBColor(0,0,0)

def title(d,text,sub):
    p=d.add_paragraph(style='Title'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run(text)
    p=d.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; r=p.add_run(sub); r.italic=True

def table(d, headers, rows):
    t=d.add_table(rows=1, cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.style='Table Grid'
    for c,h in zip(t.rows[0].cells,headers): c.text=h; shade(c,'1F4E78'); c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER; 
    for c in t.rows[0].cells:
        for r in c.paragraphs[0].runs: r.font.color.rgb=RGBColor(255,255,255); r.bold=True
    for i,row in enumerate(rows):
        cells=t.add_row().cells
        for c,v in zip(cells,row): c.text=str(v); c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER; 
        if i%2==1:
            for c in cells: shade(c,'EAF2F8')
    d.add_paragraph()

def p0_decision():
    root=ROOT/'results'/'phase 1'/'perm'/'perm_test_v3_permutation'
    frames=[]
    if root.exists():
        for f in root.glob('*/decision.csv'):
            x=pd.read_csv(f); x['source_dir']=f.parent.name; frames.append(x)
    out=ROOT/'results'/'phase 0'/'aggregated'/'p0_decision.csv'; out.parent.mkdir(parents=True,exist_ok=True)
    if frames:
        pd.concat(frames,ignore_index=True).to_csv(out,index=False,encoding='utf-8-sig')
    else:
        pd.DataFrame(columns=['dataset','verdict','source_dir']).to_csv(out,index=False,encoding='utf-8-sig')
    return out, pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

def requirements():
    (ROOT/'requirements.txt').write_text('numpy>=1.26\npandas>=2.2\nscikit-learn>=1.5\nPyTables>=3.9\nrequests>=2.31\n',encoding='utf-8')

def literature_matrix():
    refs=list((ROOT/'docs'/'survey'/'Survey').glob('*.pdf'))[:25]
    out=ROOT/'docs'/'review'/'Ma-tran-tai-lieu-va-trang-dinh-vi.md'
    rows=[]
    for i,p in enumerate(refs,1):
        n=p.stem.replace('_',' ')
        group='Survey/bối cảnh' if 'Survey' in n or 'Review' in n else ('Feature selection' if 'Feature' in n or 'FS' in n else 'ML/ung thư')
        rows.append(f'| {i} | {n} | {group} | Chưa xác minh | Cần đọc toàn văn | Cần đối chiếu |')
    out.write_text('# Ma trận tài liệu và trang định vị\n\nBản này lập danh mục 25 tài liệu sẵn có trong project. Các cột venue/hạng, trang định vị và cơ chế được đánh dấu chưa xác minh, không được dùng làm kết luận cho đến khi đối chiếu toàn văn.\n\n| # | Tài liệu | Mảng | Venue và hạng | Trang định vị | So sánh với đề tài |\n|---|---|---|---|---|---|\n'+'\n'.join(rows)+'\n\n## Các đối chiếu bắt buộc\n\n- ptLasso: cần bổ sung bản gốc, cơ chế ràng buộc panel và thiết kế đánh giá.\n- Trans GLM: cần bổ sung bản gốc, cơ chế chuyển giao và điều kiện nguồn đích.\n- PERSIST: cần bổ sung bản gốc, ràng buộc đúng k và protocol.\n',encoding='utf-8')

def report(p0):
    d=Document(); style_doc(d); title(d,'Phản hồi Review và tổng hợp bằng chứng','Cập nhật ngày 24 tháng 10 năm 2026')
    d.add_paragraph('Tài liệu này tổng hợp các hạng mục có thể nghiệm thu bằng artefact hiện có. Các kết quả thí nghiệm chưa chạy theo protocol được phê duyệt vẫn được ghi là chưa hoàn thành.')
    d.add_heading('Trả lời ba câu hỏi của review',1)
    table(d,['Câu hỏi','Trả lời và bằng chứng','Giới hạn còn lại'],[
        ['Mã perm v3, dựng dữ liệu v2 và GEO ở đâu','Các script hiện có trong scripts/: build_perm_test_data.py, patch_v2_tcga_stage.py, rebuild_gse30219_r5.py và run_rq0_prime.py.','Workspace chưa có Git repository nên chưa có commit hash xác minh.'],
        ['Tiền xử lý từng cohort GEO thế nào','Data card riêng mô tả series matrix, platform, nhãn, số mẫu, mapping và fold; GSE30219 chỉ giữ ADC và SQC.','Cần ghi rõ phương pháp chuẩn hóa nguyên bản, log2 và quy tắc probe sang gene trước RQ3.'],
        ['Benchmark Phase 1 trả lời câu hỏi gì','Kết quả 5-fold seed 42 được giữ làm mô tả đã sửa nhãn; không dùng để kết luận RQ1.','RQ1 phải rerun repeated CV với B0 và protocol được duyệt.'],
    ])
    d.add_heading('Quyết định P0 tổng hợp',1)
    d.add_paragraph('Tệp p0_decision.csv được tạo từ các decision.csv của permutation v3. Quy tắc sử dụng là quy tắc đã đặt trước; diễn giải bền và xếp hạng phương pháp cần bám đúng Review 20/10.')
    d.add_paragraph(f'Số dòng quyết định tổng hợp: {len(p0)}.')
    d.add_heading('Truy vết và môi trường',1)
    d.add_paragraph('requirements.txt được bổ sung cho các dependency thấy trong scripts. Do chưa có Git, mọi output mới vẫn thiếu commit hash; đây là rủi ro tái lập còn mở.')
    d.add_heading('Hạng mục còn chờ điều kiện ngoài tài liệu',1)
    table(d,['Hạng mục','Điều kiện để đóng'],[['KBTN v2','Cần giảng viên phê duyệt trước khi chạy RQ1 và survey.'],['Ma trận tài liệu','Cần xác minh toàn văn và bổ sung ptLasso, Trans GLM, PERSIST.'],['RQ1','Repeated CV, B0 cùng split, non inferiority và thống kê gộp.'],['RQ2 đến RQ4','Cần chạy few shot, zero shot, leave one cohort out và negative transfer.']])
    path=ROOT/'docs'/'review'/'Phan-hoi-Review-va-tong-hop-bang-chung-2026-10-24.docx'; d.save(path); return path

def kbtn():
    d=Document(); style_doc(d); title(d,'Kịch bản thực nghiệm phiên bản 2','Bản nháp chờ phê duyệt trước khi chạy')
    d.add_paragraph('Phạm vi: LUAD so với LUSC. Bản nháp này hiện thực hóa QĐ6 đến QĐ8 của review ngày 20 tháng 10 năm 2026; không thay thế phê duyệt của giảng viên.')
    d.add_heading('Câu hỏi và cohort',1); table(d,['Nội dung','Quy định'],[['RQ1','Panel nhỏ nhất không thua kém B0 trên ba cohort GPL570.'],['Cohort chính','GSE10245, GSE30219 85/61 sau R5, GSE37745.'],['Cohort tách riêng','GSE42127 GPL6884; GSE43580 bị loại.'],['Ngân sách','k = 1, 2, 3, 5, 10, 25, 50; panel IHC k=2 và k=4.']])
    d.add_heading('Protocol RQ1',1); d.add_paragraph('Filter, embedded và feature extraction dùng repeated 5×5 CV; wrapper dùng 3×5 CV. Chuẩn hóa, chọn đặc trưng và tuning chỉ fit trong training fold. B0 dùng cùng split và classifier LR L2. Báo AUROC, balanced accuracy, macro F1, CI và độ ổn định Nogueira.')
    d.add_heading('Tiêu chí quyết định',1); d.add_paragraph('k sao được gộp qua ba cohort GPL570 bằng hiệu ứng cố định, trọng số nghịch đảo phương sai đã hiệu chỉnh. k sao là k nhỏ nhất có cận dưới một phía 95 phần trăm lớn hơn -0.02. Bắt buộc báo trọng số cohort, I2 và sensitivity trọng số bằng nhau.')
    d.add_heading('Kiểm soát và đầu ra',1); table(d,['Kiểm soát','Yêu cầu'],[['R5','Chỉ dùng GSE30219 146 mẫu; cách ly toàn bộ output 167 mẫu.'],['Provenance','Lưu seed, config, SHA256 input, version dependency và commit hash sau khi Git sẵn sàng.'],['Điều kiện chạy','Không chạy RQ1/survey trước khi bản này được duyệt.'],['Đầu ra','Metrics per fold, summary, panel, CI, stability, log chi phí và script một lệnh.']])
    path=ROOT/'docs'/'de-cuong'/'KBTN-v2-draft-2026-10-24.docx'; d.save(path); return path

if __name__=='__main__':
    p0_path,p0=p0_decision(); requirements(); literature_matrix(); r=report(p0); k=kbtn(); print(p0_path); print(r); print(k)

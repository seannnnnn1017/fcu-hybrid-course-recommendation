from seleniumbase import SB
import time

class InteractiveFormFiller:
    def __init__(self, sb):
        self.sb = sb
        self.form_data = {
            'degree': '',
            'dept': '',
            'unit': '',
            'class': ''
        }
    
    def get_select_options(self, select_id):
        """取得下拉選單的所有選項"""
        try:
            # 點擊下拉選單打開選項
            self.sb.click(select_id)
 
            
            # 取得所有選項
            options = self.sb.find_elements("md-option")
            option_list = []
            
            for i, option in enumerate(options):
                try:
                    text = option.text.strip()
                    if text:
                        option_list.append({
                            'index': i,
                            'text': text,
                            'element': option
                        })
                except:
                    continue
            
            # 點擊其他地方關閉選單
            self.sb.click("body")
    
            
            return option_list
        
        except Exception as e:
            print(f"取得選項時發生錯誤: {e}")
            return []
    
    def display_options(self, options, field_name):
        """顯示選項並讓用戶選擇"""
        if not options:
            print(f"沒有找到 {field_name} 的選項")
            return None
        
        print(f"\n=== {field_name} 選項 ===")
        for i, option in enumerate(options, 1):
            print(f"{i}. {option['text']}")
        
        while True:
            try:
                choice = input(f"\n請選擇 {field_name} (輸入數字 1-{len(options)}，0=跳過): ")
                
                if choice == '0':
                    return None
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(options):
                    selected_option = options[choice_num - 1]
                    print(f"您選擇了: {selected_option['text']}")
                    return selected_option
                else:
                    print(f"請輸入 1-{len(options)} 之間的數字")
            
            except ValueError:
                print("請輸入有效的數字")
    
    def select_option(self, select_id, option):
        """選擇下拉選單選項"""
        try:
            # 點擊打開下拉選單
            self.sb.click(select_id)
            time.sleep(1)
            
            # 點擊選中的選項
            option['element'].click()
            time.sleep(1)
            
            return True
        
        except Exception as e:
            print(f"選擇選項時發生錯誤: {e}")
            return False
    
    def fill_form_interactively(self):
        """互動式填寫表單"""
        
        print("開始填寫表單...")
        
        # 1. 選擇學制
        print("\n步驟 1: 選擇學制")
        degree_select = "md-select[ng-model='searchCtrl.type1Options.degree']"
        
        try:
            degree_options = self.get_select_options(degree_select)
            selected_degree = self.display_options(degree_options, "學制")
            
            if selected_degree:
                self.select_option(degree_select, selected_degree)
                self.form_data['degree'] = selected_degree['text']
        except Exception as e:
            print(f"處理學制選項時發生錯誤: {e}")
        
        # 2. 選擇學院
        print("\n步驟 2: 選擇學院")
        dept_select = "md-select[ng-model='searchCtrl.type1Options.deptId']"
        
        try:
            # 等待學院選項載入
            time.sleep(2)
            dept_options = self.get_select_options(dept_select)
            selected_dept = self.display_options(dept_options, "學院")
            
            if selected_dept:
                self.select_option(dept_select, selected_dept)
                self.form_data['dept'] = selected_dept['text']
                time.sleep(2)  # 等待系所選項載入
        except Exception as e:
            print(f"處理學院選項時發生錯誤: {e}")
        
        # 3. 選擇系所
        print("\n步驟 3: 選擇系所")
        unit_select = "md-select[ng-model='searchCtrl.type1Options.unitId']"
        
        try:
            unit_options = self.get_select_options(unit_select)
            selected_unit = self.display_options(unit_options, "系所")
            
            if selected_unit:
                self.select_option(unit_select, selected_unit)
                self.form_data['unit'] = selected_unit['text']
                time.sleep(2)  # 等待班級選項載入
        except Exception as e:
            print(f"處理系所選項時發生錯誤: {e}")
        
        # 4. 選擇班級
        print("\n步驟 4: 選擇班級")
        class_select = "md-select[ng-model='searchCtrl.type1Options.classId']"
        
        try:
            class_options = self.get_select_options(class_select)
            selected_class = self.display_options(class_options, "班級")
            
            if selected_class:
                self.select_option(class_select, selected_class)
                self.form_data['class'] = selected_class['text']
        except Exception as e:
            print(f"處理班級選項時發生錯誤: {e}")
        
        # 5. 確認並送出
        self.confirm_and_submit()
    
    def confirm_and_submit(self):
        """確認表單內容並送出"""
        print("\n=== 表單內容確認 ===")
        print(f"學制: {self.form_data['degree']}")
        print(f"學院: {self.form_data['dept']}")
        print(f"系所: {self.form_data['unit']}")
        print(f"班級: {self.form_data['class']}")
        
        while True:
            confirm = input("\n確認送出表單嗎? (y/n): ").lower()
            
            if confirm == 'y':
                try:
                    # 點擊查詢按鈕
                    search_button = "button[ng-click='searchCtrl.searchType1()']"
                    self.sb.click(search_button)
                    print("表單已送出!")
                    return True
                
                except Exception as e:
                    print(f"送出表單時發生錯誤: {e}")
                    return False
            
            elif confirm == 'n':
                print("取消送出表單")
                return False
            
            else:
                print("請輸入 y 或 n")

class CourseDataScraper:
    def __init__(self, sb):
        self.sb = sb
        self.course_data = []
        self.discriptGate = True  # 用於控制是否獲取課程描述
    
    def get_course_description(self, row, index):
        """點進課程連結，切換分頁後獲取課程描述並關閉分頁"""
        try:
            print(f"  正在獲取第 {index} 筆課程描述...")

            # 找到課程名稱的連結
            course_name_cell = row.find_element("xpath", ".//td[@data-title='課程名稱']")
            course_link = course_name_cell.find_element("xpath", ".//a")

            # 紀錄目前的分頁
            original_window = self.sb.driver.current_window_handle
            existing_windows = self.sb.driver.window_handles

            # 用 JavaScript 點擊連結（防止阻擋）
            self.sb.execute_script("arguments[0].click();", course_link)

            # 等待新分頁出現
            timeout = 5
            start_time = time.time()
            while time.time() - start_time < timeout:
                new_windows = self.sb.driver.window_handles
                if len(new_windows) > len(existing_windows):
                    break
                time.sleep(0.2)
            else:
                print("    新分頁未出現，跳過")
                return "無法開啟課程描述分頁"

            # 切換到新分頁
            new_windows = self.sb.driver.window_handles
            new_tab = [w for w in new_windows if w not in existing_windows]
            if not new_tab:
                print("    找不到新分頁，跳過此課程")
                return "無法開啟課程描述分頁"
            self.sb.driver.switch_to.window(new_tab[0])

            # 等待頁面載入
            time.sleep(2)

            description = ""
            try:
                # 等待課程描述區域載入
                self.sb.wait_for_element("h3:contains('課程描述')", timeout=10)

                # 嘗試擷取課程描述
                description_elements = self.sb.find_elements("div[style*='background-color:#e5e5e56e']")
                if description_elements:
                    description = description_elements[0].text.strip()
                else:
                    # 備案：嘗試找其他容器
                    description_divs = self.sb.find_elements("div.ng-binding")
                    for div in description_divs:
                        text = div.text.strip()
                        if len(text) > 20:
                            description = text
                            break

                if not description:
                    description = "無課程描述"

            except Exception as desc_e:
                print(f"    獲取課程描述時發生錯誤: {desc_e}")
                description = "獲取課程描述失敗"

            # 關閉該分頁
            self.sb.driver.close()

            # 切回原本的分頁
            self.sb.driver.switch_to.window(original_window)

            # 等待課程列表重新載入
            self.sb.wait_for_element("tr[ng-repeat*='item in searchCtrl.searchResult.items']", timeout=10)
            time.sleep(1)

            return description

        except Exception as e:
            print(f"    點擊第 {index} 筆課程連結時發生錯誤: {e}")
            return "獲取課程描述失敗"

    def scrape_course_data(self, max_courses=None):
        """爬取課程資料"""
        
        print("正在等待查詢結果載入...")
        
        try:
            # 等待表格資料載入
            self.sb.wait_for_element("tr[ng-repeat*='item in searchCtrl.searchResult.items']", timeout=15)
            time.sleep(3)
            
            # 取得所有課程行
            course_rows = self.sb.find_elements("tr[ng-repeat*='item in searchCtrl.searchResult.items']")
            
            if not course_rows:
                print("沒有找到課程資料")
                return False
            
            total_courses = len(course_rows)
            print(f"找到 {total_courses} 筆課程資料")
            
            # 決定要爬取的課程數量
            if max_courses is None:
                courses_to_scrape = total_courses
            else:
                courses_to_scrape = min(max_courses, total_courses)
            
            print(f"開始爬取前 {courses_to_scrape} 筆課程資料...")
            
            # 爬取每筆課程資料
            for i, row in enumerate(course_rows[:courses_to_scrape]):
                try:
                    course_info = self.extract_course_info(row, i + 1)
                    if course_info:
                        self.course_data.append(course_info)
                        print(f"已爬取第 {i + 1} 筆: {course_info['課程名稱']}")
                
                except Exception as e:
                    print(f"爬取第 {i + 1} 筆資料時發生錯誤: {e}")
                    continue
            
            print(f"成功爬取 {len(self.course_data)} 筆課程資料")
            return True
        
        except Exception as e:
            print(f"爬取課程資料時發生錯誤: {e}")
            return False
    
    def extract_course_info(self, row, index):
        """從單一課程行中提取資訊"""
        
        try:
            # 取得所有 td 元素
            cells = row.find_elements("xpath", ".//td")
            
            if len(cells) < 8:
                print(f"第 {index} 筆資料欄位不足")
                return None
            
            course_info = {}
            
            # 根據 HTML 結構提取資料
            # 注意：需要跳過隱藏的按鈕欄位
            
            # 找到包含選課代碼的 td (bo-text="item.scr_selcode")
            course_code_cell = row.find_element("xpath", ".//td[@bo-text='item.scr_selcode']")
            course_info['選課代碼'] = course_code_cell.text.strip()
            
            # 找到包含課程編碼的 td (bo-text="item.sub_id3")
            course_id_cell = row.find_element("xpath", ".//td[@bo-text='item.sub_id3']")
            course_info['課程編碼'] = course_id_cell.text.strip()
            
            # 找到包含課程名稱的 td (包含 a 標籤)
            course_name_cell = row.find_element("xpath", ".//td[@data-title='課程名稱']")
            course_name_link = course_name_cell.find_element("xpath", ".//a")
            course_info['課程名稱'] = course_name_link.text.strip()
            
            # 找到包含學分的 td (bo-text="item.scr_credit")
            credit_cell = row.find_element("xpath", ".//td[@bo-text='item.scr_credit']")
            course_info['學分'] = credit_cell.text.strip()
            
            # 找到包含必選修的 td (bo-text="item.scj_scr_mso")
            required_cell = row.find_element("xpath", ".//td[@bo-text='item.scj_scr_mso']")
            course_info['必選修'] = required_cell.text.strip()
            
            # 找到包含上課方式的 td (bo-text="item.scr_ldl")
            method_cell = row.find_element("xpath", ".//td[@bo-text='item.scr_ldl']")
            course_info['上課方式'] = method_cell.text.strip()
            if self.discriptGate:
                get_course_desc = self.get_course_description(row, index)
                course_info['課程描述'] = get_course_desc
            return course_info
        
        except Exception as e:
            print(f"提取第 {index} 筆課程資訊時發生錯誤: {e}")
            return None
    
    def save_to_csv(self, filename=None):
        """將課程資料儲存為 CSV 檔案"""
        
        if not self.course_data:
            print("沒有課程資料可以儲存")
            return False
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"課程資料_{timestamp}.csv"
        
        try:
            # CSV 欄位順序
            fieldnames = ['選課代碼', '課程編碼', '課程名稱', '學分', '必選修', '上課方式','課程描述']
            
            with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # 寫入標題行
                writer.writeheader()
                
                # 寫入課程資料
                for course in self.course_data:
                    writer.writerow(course)
            
            print(f"課程資料已成功儲存到: {filename}")
            print(f"共儲存 {len(self.course_data)} 筆課程資料")
            return True
        
        except Exception as e:
            print(f"儲存 CSV 檔案時發生錯誤: {e}")
            return False
    
    def display_scraped_data(self):
        """顯示已爬取的課程資料"""
        
        if not self.course_data:
            print("沒有課程資料")
            return
        
        print(f"\n=== 已爬取的課程資料 ({len(self.course_data)} 筆) ===")
        for i, course in enumerate(self.course_data, 1):
            print(f"\n第 {i} 筆:")
            print(f"  選課代碼: {course['選課代碼']}")
            print(f"  課程編碼: {course['課程編碼']}")
            print(f"  課程名稱: {course['課程名稱']}")
            print(f"  學分: {course['學分']}")
            print(f"  必選修: {course['必選修']}")
            print(f"  上課方式: {course['上課方式']}")

import csv
from datetime import datetime


def main_with_scraper():
    """包含課程爬取功能的主程式"""
    
    print("=== 課程查詢與資料爬取工具 ===")
    
    # 請用戶輸入目標網址
    url = 'https://coursesearch02.fcu.edu.tw/main.aspx?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE3NDkwMTA3ODJ9.0CC2LkjPP4H17yEMK0nsQvJtxmrMHxmSbZgNSn_R9Yk'
    
    if not url:
        print("未輸入網址，程式結束")
        return
    
    try:
        with SB(uc=True, headless=False, slow=True) as sb:
            
            print(f"正在開啟網頁: {url}")
            sb.open(url)
            
            # 等待頁面載入
            sb.wait_for_element("md-card", timeout=10)
            
            # 1. 填寫表單
            print("\n步驟 1: 填寫查詢表單")
            form_filler = InteractiveFormFiller(sb)
            form_filler.fill_form_interactively()
            
            # 等待查詢結果載入
            print("\n步驟 2: 等待查詢結果...")
   
            
            # 2. 爬取課程資料
            print("\n步驟 3: 爬取課程資料")
            
            # 詢問用戶要爬取多少筆資料
            while True:
                try:
                    max_courses_input = input("請輸入要爬取的課程數量 (直接按 Enter 表示全部爬取): ").strip()
                    
                    if max_courses_input == "":
                        max_courses = None
                        break
                    else:
                        max_courses = int(max_courses_input)
                        if max_courses > 0:
                            break
                        else:
                            print("請輸入大於 0 的數字")
                
                except ValueError:
                    print("請輸入有效的數字")
            
            # 開始爬取
            scraper = CourseDataScraper(sb)
            
            if scraper.scrape_course_data(max_courses):
                
                # 3. 顯示爬取結果
                scraper.display_scraped_data()
                
                # 4. 儲存為 CSV
                save_choice = input("\n是否要儲存為 CSV 檔案? (y/n): ").lower()
                
                if save_choice == 'y':
                    custom_filename = input("請輸入檔案名稱 (直接按 Enter 使用預設名稱): ").strip()
                    filename = custom_filename if custom_filename else None
                    scraper.save_to_csv(filename)
            
            # 等待用戶查看結果
            input("\n按 Enter 鍵結束程式...")
    
    except Exception as e:
        print(f"程式執行過程中發生錯誤: {e}")

if __name__ == "__main__":

    main_with_scraper()

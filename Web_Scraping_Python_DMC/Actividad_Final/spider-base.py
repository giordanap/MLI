import scrapy
import logging
from selenium.webdriver import Chrome
from ripley.items import ComputerItem

class RipleySpiderSpider(scrapy.Spider):
    name = 'ripley_spider'
    allowed_domains = []
    start_urls = []
    custom_settings = {
    'LOG_LEVEL': logging.WARNING,
    'ITEM_PIPELINES': {'ripley.pipelines.RipleyPipeline': 1},
    'FEED_FORMAT':'csv',
    'FEED_URI': 'computers.csv',
    'DOWNLOAD_DELAY':0.5
    }
    
    def __init__(self,*args,**kwargs):
        # COMPLETAR - guardar al driver de Chrome en la variable browser
		
    def set_selectors(self):
        # Para navegar (XPATH)
        self.get_all_items = ''
        self.get_next_page = ''
        # Para extraer la informacion (CSS)
        self.get_name = ''
        self.get_sku = ''
        self.get_description = ''
        self.get_img_link = ''

    def parse(self, response):
        self.browser.get(response.url)
        self.browser.implicitly_wait(10)
        # COMPLETAR para llegar de la página principal a la categoría elegida
        url = self.browser.current_url
        yield scrapy.Request(url=url,callback=self.parse_page,dont_filter=True)
            
    def parse_page(self, response):
        # COMPLETAR - CARGAR PAGINA Y BUSCAR TODOS LOS LINKS A ITEMS
        next_page_href = self.browser.find_element_by_xpath(self.get_next_page)
        if next_page_href:
            next_url = # COMPLETAR
            yield scrapy.Request(url=next_url,callback=self.parse_page,dont_filter=True)
    
    def parse_item(self, response):
        item = ComputerItem()
        # COMPLETAR - ALMACENAR INFORMACION
        return item